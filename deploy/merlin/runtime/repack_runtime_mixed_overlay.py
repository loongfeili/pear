#!/usr/bin/env python3
"""Derive an unpublished runtime bundle from an immutable native parent archive.

Only the reviewed GDN JIT and rollout allocator Python overlays are supported. No installs, imports from
the runtime, source edits, extraction, GPU work, or global READY publication occur.
Run only after the FMA regression and native allocator CUDA IPC probe pass;
full restored-environment and actual GRPO validation are still required.
"""
import argparse
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import secrets
import subprocess
import sys
import tarfile
import time

ROOT = Path("/mnt/bn/ic-vlm/personal/loongfei.li03/pear_20261114")
PREFIX = "/opt/tiger/pear_runtime_20261114"
COMMIT = "f32aa1f7c2ec9c3f45630cc9f8541b3201e8e087"
CUDA = "pear/models/qwen3_5/rollout/hopper/csrc/gdn_chunk.cu"
BEFORE = "8437dda87bc17adc16f229304ef34a5a78b956d145881213fa362192d86d1343"
AFTER = "ad089ddfc5b1ca965a17f031e44f12bd8dd08bceb17db108107eda459b09dc85"
PATCH_SHA = "6cdc0a464f07ca4673dfaf269d7728122e3d3bd45d75bcefe0672a513082db3e"
PYTHON_SOURCE = "pear/rollout/server.py"
PYTHON_BEFORE = "8e1df8c596aa515a4ed332169213f323246fb3e8fda126320a4f8cb617081710"
PYTHON_AFTER = "3967a23d087bb2039f4105dd7cd31cf8e10cb02199c5d71850b2ed51e27b306d"
ALLOCATOR_PATCH_SHA = "89ed0a955cc6fe7aded03e48c46c5d090ed596962b46699613ea9c9ce57d5461"
NATIVE_BUNDLE = "runtime-20261114T123450Z-3681"
PREVIOUS_BUNDLE = "runtime-20261114T141008Z-jit-fma-e59b7570b9cc"
PREVIOUS_ARCHIVE_SHA = "2118994b29f4705c47112d3c3a79fb841970f1b1cfe8a52e5ef1ab3aed607029"
PREVIOUS_SOURCE_SHA = "b81dabd71411ef34880917ebc51643702b7e4b71ee6aa828ba2d032b432d7701"
MAX_MEMBERS = 1_000_000
MAX_FILE = 16 * 2**30
MAX_TOTAL = 512 * 2**30
MAX_JSON = 4 * 2**20
IDENTITY = "source_identity.json"
PROVENANCE = "jit_overlay_provenance.json"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with path.open("rb") as stream:
        return stream_sha(stream)


def stream_sha(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(4 * 2**20), b""):
        result.update(block)
    return result.hexdigest()


def hex_sha(value):
    require(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value), "Invalid SHA256")
    return value


def bounded_file(value, base=ROOT, maximum=MAX_JSON):
    path = Path(value).resolve(strict=True)
    require(path.is_relative_to(base.resolve(strict=True)) and path.is_file(), f"File outside allowed root: {path}")
    require(path.stat().st_size <= maximum, f"Input too large: {path}")
    return path


def read_json(path):
    raw = path.read_bytes()
    require(len(raw) <= MAX_JSON, f"JSON too large: {path}")
    return raw, json.loads(raw)


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def identity_ok(value):
    require(isinstance(value, dict) and set(value) == {"status", "commit", "source_sha256"}, "Unexpected source identity schema")
    require(value["status"] == "source_only_verified" and value["commit"] == COMMIT, "Unexpected source identity")
    hex_sha(value["source_sha256"])


def relative_name(name):
    require(isinstance(name, str) and 0 < len(name) <= 4096 and "\x00" not in name and "\\" not in name,
            "Invalid archive/source path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts, f"Unsafe path: {name}")
    clean = str(path)
    require(clean not in ("", "."), f"Empty path: {name}")
    return clean


def source_snapshot(repo, manifest, parent_identity, effective_identity, previous_identity):
    require(manifest.get("upstream_commit") == COMMIT, "Unexpected source manifest commit")
    files = manifest["files"]
    require(isinstance(files, dict) and 0 < len(files) <= 10_000 and files.get(CUDA) == BEFORE and files.get(PYTHON_SOURCE) == PYTHON_BEFORE,
            "Unexpected source manifest file set")
    actual = {}
    for name, original in files.items():
        require(relative_name(name) == name, f"Noncanonical source path: {name}")
        path = bounded_file(repo / name, repo, 512 * 2**20)
        patched = manifest.get("allowed_patched_files", {}).get(name, [])
        allowed = {hex_sha(original), *(patched if isinstance(patched, list) else [patched])}
        actual[name] = file_sha(path)
        require(actual[name] in allowed, f"Source not approved by manifest: {name}")
    require(actual[CUDA] == AFTER, "Live CUDA source is not the reviewed FMA candidate")
    require(actual[PYTHON_SOURCE] == PYTHON_AFTER, "Live server source is not the reviewed allocator patch")
    aggregate = lambda values: sha(json.dumps(values, sort_keys=True).encode())
    require(aggregate(actual) == effective_identity["source_sha256"], "Effective source hash differs")
    actual[PYTHON_SOURCE] = PYTHON_BEFORE
    require(aggregate(actual) == previous_identity["source_sha256"],
            "Other tracked source changed since the previous FMA overlay")
    actual[CUDA] = BEFORE
    require(aggregate(actual) == parent_identity["source_sha256"],
            "Other tracked source changed since the native parent build")
    return len(files)


def validate_member(member, seen, non_directories):
    name = relative_name(member.name)
    require(name not in seen and len(seen) < MAX_MEMBERS, f"Duplicate/excess archive entry: {name}")
    require(name.split("/", 1)[0] in {"envs", "bin", ".pear_local_runtime", IDENTITY}, f"Unexpected parent entry: {name}")
    if name.split("/", 1)[0] in {".pear_local_runtime", IDENTITY}:
        require(name in {".pear_local_runtime", IDENTITY} and member.isreg(), f"Unexpected metadata entry: {name}")
    require(member.isdir() or member.isreg() or member.issym() or member.islnk(), f"Unsupported archive entry: {name}")
    require(not member.sparse and 0 <= member.size <= MAX_FILE, f"Sparse/oversized entry: {name}")
    require(member.isreg() or member.size == 0, f"Nonregular entry has payload: {name}")
    require(all(str(p) not in non_directories for p in PurePosixPath(name).parents), f"Entry traverses a non-directory: {name}")
    if member.issym() or member.islnk():
        target = member.linkname
        require(target and len(target) <= 4096 and "\x00" not in target and "\\" not in target, f"Invalid link: {name}")
        if member.islnk():
            target = relative_name(target)
            require(target in seen and seen[target] in {"file", "hardlink"}, f"Hardlink has no prior regular target: {name}")
        elif target.startswith("/"):
            require(target.startswith(PREFIX + "/"), f"Symlink leaves runtime prefix: {name}")
            target = posixpath.normpath(target[len(PREFIX) + 1:])
        else:
            target = posixpath.normpath(posixpath.join(posixpath.dirname(name), target))
        require(target.split("/", 1)[0] in {"envs", "bin"}, f"Link leaves envs/bin: {name}")
    kind = "dir" if member.isdir() else "file" if member.isreg() else "symlink" if member.issym() else "hardlink"
    seen[name] = kind
    if kind != "dir":
        non_directories.add(name)
    return name, kind


def atomic_file(path, data):
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def file_signature(stream):
    status = os.fstat(stream.fileno())
    return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effective-source-identity", required=True, help="Reviewed --check-source-only JSON, inside task ROOT")
    parser.add_argument("--expected-effective-source-sha256", required=True)
    parser.add_argument("--patch", required=True, help="Reviewed h20_gdn_fma_order.patch, inside task ROOT")
    parser.add_argument("--allocator-patch", required=True, help="Reviewed rollout-only native allocator patch, inside task ROOT")
    parser.add_argument("--parent-ready", default=str(ROOT / "runtime_bundles" / NATIVE_BUNDLE / "READY"))
    parser.add_argument("--previous-ready", default=str(ROOT / "runtime_bundles" / PREVIOUS_BUNDLE / "READY"))
    parser.add_argument("--expected-parent-ready-sha256", required=True, help="SHA256 of exact chosen READY file bytes")
    parser.add_argument("--source-manifest", default=str(ROOT / "launch/source_manifest.json"))
    args = parser.parse_args()
    bundles = (ROOT / "runtime_bundles").resolve(strict=True)
    require(bundles == ROOT / "runtime_bundles", "Bundle root must not be a symlink")
    parent_ready = bounded_file(args.parent_ready, bundles)
    ready_raw, parent = read_json(parent_ready)
    require(sha(ready_raw) == hex_sha(args.expected_parent_ready_sha256), "Parent READY bytes changed")
    require(re.fullmatch(r"runtime-[a-zA-Z0-9_-]+", parent["bundle"]) and parent["archive"] == "runtime.tar.gz",
            "Unexpected parent bundle/archive name")
    require(parent["prefix"] == PREFIX and isinstance(parent["size"], int) and 0 < parent["size"] <= MAX_TOTAL,
            "Unexpected parent prefix/size")
    hex_sha(parent["sha256"])
    identity_ok(parent["source_identity"])
    require(parent["bundle"] == NATIVE_BUNDLE, "Unexpected native parent bundle")
    require("jit_overlay_provenance" not in parent, "This builder accepts only a native parent, not a prior derived bundle")
    parent_dir = bundles / parent["bundle"]
    require(parent_dir.resolve(strict=True) == parent_dir, "Parent bundle directory must not be a symlink")
    per_bundle_ready = bounded_file(parent_dir / "READY", parent_dir)
    require(read_json(per_bundle_ready)[1] == parent, "Parent per-bundle READY differs")
    archive = bounded_file(parent_dir / parent["archive"], parent_dir, MAX_TOTAL)
    require(archive.stat().st_size == parent["size"], "Parent archive size differs")
    previous_dir = bundles / PREVIOUS_BUNDLE
    require(previous_dir.resolve(strict=True) == previous_dir, "Previous bundle directory must not be a symlink")
    previous_ready = bounded_file(args.previous_ready, previous_dir)
    previous_raw, previous = read_json(previous_ready)
    require(previous["bundle"] == PREVIOUS_BUNDLE and previous["sha256"] == PREVIOUS_ARCHIVE_SHA and
            previous["archive"] == "runtime.tar.gz" and previous["prefix"] == PREFIX,
            "Unexpected preceding effective bundle")
    identity_ok(previous["source_identity"])
    require(previous["source_identity"]["source_sha256"] == PREVIOUS_SOURCE_SHA,
            "Unexpected preceding effective source identity")
    require(previous.get("jit_overlay_provenance", {}).get("parent_bundle") == parent["bundle"] and
            previous["jit_overlay_provenance"]["parent_archive_sha256"] == parent["sha256"],
            "Previous overlay does not point to this native parent")
    allocator_patch = bounded_file(args.allocator_patch)
    require(file_sha(allocator_patch) == ALLOCATOR_PATCH_SHA, "Allocator patch differs from reviewed patch")
    patch = bounded_file(args.patch)
    require(file_sha(patch) == PATCH_SHA, "Patch is not the reviewed minimal FMA patch")
    effective_path = bounded_file(args.effective_source_identity)
    effective_raw, effective = read_json(effective_path)
    identity_ok(effective)
    require(effective["source_sha256"] == hex_sha(args.expected_effective_source_sha256), "Effective identity differs from reviewed hash")
    manifest_path = bounded_file(args.source_manifest)
    manifest_raw, manifest = read_json(manifest_path)
    checker = bounded_file(ROOT / "launch/preflight_stage1.py")
    checker_sha = file_sha(checker)
    repo = (ROOT / "repo").resolve(strict=True)
    require(repo == ROOT / "repo", "Repository root must not be a symlink")
    tracked_count = source_snapshot(repo, manifest, parent["source_identity"], effective, previous["source_identity"])
    checked = subprocess.check_output([sys.executable, "-B", str(checker), "--root", str(ROOT),
                                      "--source-manifest", str(manifest_path), "--check-source-only"], timeout=180)
    require(len(checked) <= MAX_JSON and json.loads(checked) == effective, "Actual source checker disagrees with reviewed identity")
    checked = checked.rstrip() + b"\n"

    # Pin and hash the actual immutable parent before creating any output directory.
    with archive.open("rb") as original:
        parent_signature = file_signature(original)
        require(stream_sha(original) == parent["sha256"], "Parent archive SHA256 differs")
        require(file_signature(original) == parent_signature, "Parent archive changed while hashing")
        original.seek(0)
        created = time.time()
        bundle = "runtime-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-mixed-source-" + secrets.token_hex(6)
        destination = bundles / bundle
        destination.mkdir(mode=0o700)
        partial = destination / "runtime.tar.gz.partial"
        counts = {"members": 0, "files": 0, "dirs": 0, "symlinks": 0, "hardlinks": 0, "regular_bytes": 0}
        seen, non_directories = {}, set()
        identity_seen = marker_seen = False
        try:
            with partial.open("xb") as output:
                with gzip.GzipFile(fileobj=output, filename="", mode="wb", compresslevel=1, mtime=int(created)) as compressed:
                    with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as target:
                        with tarfile.open(fileobj=original, mode="r|gz") as source:
                            for member in source:
                                name, kind = validate_member(member, seen, non_directories)
                                counts["members"] += 1
                                counts[kind + "s"] += 1
                                counts["regular_bytes"] += member.size if member.isreg() else 0
                                require(counts["regular_bytes"] <= MAX_TOTAL, "Archive logical payload exceeds bound")
                                if name == IDENTITY:
                                    require(member.size <= MAX_JSON, "Parent source identity is too large")
                                    require(json.loads(source.extractfile(member).read()) == parent["source_identity"],
                                            "Archived source identity differs from parent READY")
                                    replacement = copy.copy(member)
                                    replacement.pax_headers = dict(member.pax_headers)
                                    replacement.pax_headers.pop("size", None)
                                    replacement.size = len(checked)
                                    target.addfile(replacement, io.BytesIO(checked))
                                    identity_seen = True
                                elif name == ".pear_local_runtime":
                                    require(member.size <= 1024, "Unexpected runtime marker size")
                                    marker = source.extractfile(member).read()
                                    require(marker == b"pear-local-runtime-20261114\n", "Unexpected runtime marker")
                                    target.addfile(member, io.BytesIO(marker))
                                    marker_seen = True
                                elif member.isreg():
                                    with source.extractfile(member) as content:
                                        target.addfile(member, content)
                                else:
                                    target.addfile(member)
                        require(identity_seen and marker_seen and all(seen.get(p) == "dir" for p in (
                            "envs", "bin", "envs/verl", "envs/libero", "envs/mesa", "envs/numactl")), "Required runtime entries missing")
                        require(counts["files"] > 2 and any(n.startswith("bin/") for n in seen), "Empty native environment/tools")
                        provenance = {
                            "kind": "mixed_source_overlay_v2", "created_at": created,
                            "parent": {"bundle": parent["bundle"], "archive": parent["archive"], "sha256": parent["sha256"],
                                       "size": parent["size"], "ready_file_sha256": sha(ready_raw)},
                            "native_build_source_identity": parent["source_identity"], "effective_source_identity": effective,
                            "previous_effective_bundle": {"bundle": previous["bundle"], "sha256": previous["sha256"],
                                                          "ready_file_sha256": sha(previous_raw),
                                                          "source_identity": previous["source_identity"],
                                                          "payload_used_for_repack": False},
                            "overlays": [
                                {"kind": "cuda_jit_source", "source_path": CUDA, "before_sha256": BEFORE, "after_sha256": AFTER,
                                 "patch_path": str(patch.relative_to(ROOT)), "patch_sha256": PATCH_SHA},
                                {"kind": "python_runtime_source", "source_path": PYTHON_SOURCE,
                                 "before_sha256": PYTHON_BEFORE, "after_sha256": PYTHON_AFTER,
                                 "patch_path": str(allocator_patch.relative_to(ROOT)), "patch_sha256": ALLOCATOR_PATCH_SHA}],
                            "verification_inputs": {"checker_sha256": checker_sha, "source_manifest_sha256": sha(manifest_raw),
                                                    "reviewed_identity_file_sha256": sha(effective_raw),
                                                    "actual_checker_stdout_sha256": sha(checked), "tracked_files": tracked_count},
                            "copied_parent_archive_counts": counts,
                            "native_payload": "All envs/ and bin/ file bytes, links and logical tar metadata are copied unchanged from the parent archive.",
                            "changes": ["Replace source_identity.json with verified effective CUDA-JIT and Python source identity", "Add jit_overlay_provenance.json"],
                            "limitations": "No native dependency rebuild, GPU test, runtime restore, readiness certification or global publication is performed by this repack.",
                        }
                        provenance_raw = json_bytes(provenance)
                        info = tarfile.TarInfo(PROVENANCE)
                        info.size, info.mode, info.mtime = len(provenance_raw), 0o644, int(created)
                        target.addfile(info, io.BytesIO(provenance_raw))
                output.flush()
                os.fsync(output.fileno())
            require(file_signature(original) == parent_signature, "Parent archive changed while repacking")
            require(parent_ready.read_bytes() == ready_raw and read_json(per_bundle_ready)[1] == parent, "Parent READY changed while repacking")
            require(manifest_path.read_bytes() == manifest_raw and file_sha(checker) == checker_sha and file_sha(patch) == PATCH_SHA and file_sha(allocator_patch) == ALLOCATOR_PATCH_SHA and
                    previous_ready.read_bytes() == previous_raw,
                    "Reviewed input changed while repacking")
            source_snapshot(repo, manifest, parent["source_identity"], effective, previous["source_identity"])
            archive_sha, archive_size = file_sha(partial), partial.stat().st_size
            partial.rename(destination / "runtime.tar.gz")
            atomic_file(destination / "SHA256SUMS", (archive_sha + "  runtime.tar.gz\n").encode())
            atomic_file(destination / PROVENANCE, provenance_raw)
            ready = {"bundle": bundle, "archive": "runtime.tar.gz", "sha256": archive_sha, "size": archive_size,
                     "prefix": PREFIX, "source_identity": effective, "created_at": created,
                     "jit_overlay_provenance": {"file": PROVENANCE, "sha256": sha(provenance_raw),
                                                "parent_bundle": parent["bundle"], "parent_archive_sha256": parent["sha256"],
                                                "output_archive_members": counts["members"] + 1}}
            atomic_file(destination / "READY", json_bytes(ready))
            print(json.dumps({"status": "derived_bundle_ready_unpublished", "directory": str(destination),
                              "sha256": archive_sha, "size": archive_size, "parent_counts": counts,
                              "global_ready_changed": False}, sort_keys=True))
        except BaseException:
            # Keep the uniquely named partial output for diagnosis; never publish it.
            print(f"REPACK FAILED; unpublished output retained at {destination}", file=sys.stderr)
            raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"REPACK FAILED: {exc}", file=sys.stderr)
        sys.exit(1)

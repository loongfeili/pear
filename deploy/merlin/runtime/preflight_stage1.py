#!/usr/bin/env python3
"""Gate approved H20 Stage 1 on a successful eight-GPU preflight. Never installs."""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

COMMIT = "f32aa1f7c2ec9c3f45630cc9f8541b3201e8e087"
ROOT = "/mnt/bn/ic-vlm/personal/loongfei.li03/pear_20261114"
HELPER_DIR = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def command(args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs).strip()


def check_source(repo, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    require(manifest["upstream_commit"] == COMMIT, "Unexpected upstream source manifest")
    actual = {}
    for name, original in manifest["files"].items():
        path = repo / name
        patched = manifest.get("allowed_patched_files", {}).get(name, [])
        allowed = {original, *(patched if isinstance(patched, list) else [patched])}
        require(path.is_file(), f"Source file missing: {name}")
        actual[name] = digest(path)
        require(actual[name] in allowed, f"Source differs from reviewed manifest: {name}")
    # Main source comes from git archive; no .git directory is required.
    return hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()


def check_hardware(compat, min_memory_gib):
    rows = [list(map(str.strip, s.split(","))) for s in command([
        "nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"
    ]).splitlines()]
    require(len(rows) == 8 and all("H20" in row[0] for row in rows), "Expected eight H20 GPUs for the approved candidate")
    require(all(float(row[2]) >= 79000 for row in rows), "GPU memory below the Hopper recipe's 80 GB requirement")
    # Query selected user-space library, not nvidia-smi's kernel-driver version.
    cuda = ctypes.CDLL(str(compat / "libcuda.so.1"))
    version, count = ctypes.c_int(), ctypes.c_int()
    require(cuda.cuDriverGetVersion(ctypes.byref(version)) == 0 and version.value >= 13000,
            f"CUDA 13 driver API required; reported {version.value}")
    require(cuda.cuInit(0) == 0, "CUDA compatibility initialization failed")
    require(cuda.cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value == 8, "CUDA must expose eight assigned devices")
    for device in range(count.value):
        major, minor = ctypes.c_int(), ctypes.c_int()
        require(cuda.cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), device) == 0,
                f"Cannot query CUDA capability of GPU {device}")
        require((major.value, minor.value) == (9, 0), f"GPU {device} does not support the SM90 recipe")
    mem_kib = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemTotal:")))
    effective_memory = mem_kib * 1024
    for path in (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        if path.is_file() and path.read_text().strip().isdigit():
            effective_memory = min(effective_memory, int(path.read_text()))
    require(effective_memory >= min_memory_gib * 2**30,
            f"Effective RAM {effective_memory / 2**30:.1f} GiB below {min_memory_gib}; candidate uses 2 x 600 GB LMCache + 64 GiB Ray")
    cpu_count = len(os.sched_getaffinity(0))
    quota = Path("/sys/fs/cgroup/cpu.max")
    if quota.is_file():
        limit, period = quota.read_text().split()
        if limit != "max":
            cpu_count = min(cpu_count, int(limit) / int(period))
    require(cpu_count >= 120, "At least 120 effective CPU cores required for advertised Ray resources")
    shm = os.statvfs("/dev/shm")
    require(shm.f_bavail * shm.f_frsize >= 64 * 2**30, "At least 64 GiB available /dev/shm required")
    return {"gpu": rows, "cuda_driver_api": version.value, "memory_gib": round(effective_memory / 2**30, 1), "cpu_count": cpu_count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--repo", help="Override repository path; useful for CPU-only source verification")
    parser.add_argument("--source-manifest", default=str(HELPER_DIR / "source_manifest.json"))
    parser.add_argument("--model-dir", help="Prepared base-model directory; defaults to WORK_DIR/models/Qwen3.8-27B")
    parser.add_argument("--certify", action="store_true", help="Record identity immediately AFTER preflight_run.sh creates PASSED")
    parser.add_argument("--check-source-only", action="store_true", help="CPU-only source check; never creates readiness certificate")
    parser.add_argument("--min-memory-gib", type=int, default=1700)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    repo = Path(args.repo).resolve() if args.repo else root / "repo"
    source_hash = check_source(repo, Path(args.source_manifest))
    if args.check_source_only:
        require(not args.certify, "--check-source-only cannot certify readiness")
        print(json.dumps({"status": "source_only_verified", "commit": COMMIT, "source_sha256": source_hash}))
        return
    work = root / "work_dir"
    certificate = work / "stage1_environment_ready.json"
    if args.certify:
        certificate.unlink(missing_ok=True)
    log_dir = root / "preflight/logs"
    passed, failed = log_dir / "PASSED", log_dir / "FAILED"
    require(passed.is_file() and passed.read_text().startswith("PEAR_H20_PREFLIGHT_PASSED "), "Eight-GPU preflight PASSED marker missing")
    require(not failed.exists(), "Preflight FAILED marker remains; a clean successful preflight is required")
    log_hashes = {}
    for name in ("libero_run.log", "h20_smoke.log", "h20_acceptance.log", "single_node_grpo_smoke.log",
                 "upstream_gpu_suite.xml", "h20_compatibility_result.json", "gdn_regression_path.txt", "grpo_result_path.txt"):
        log = log_dir / name
        require(log.is_file() and log.stat().st_size > 0, f"Preflight evidence missing: {log}")
        require(log.stat().st_mtime <= passed.stat().st_mtime, f"{name} changed after PASSED; successful preflight must finish first")
        log_hashes[name] = digest(log)
    compatibility = json.loads((log_dir / "h20_compatibility_result.json").read_text())
    require(compatibility.get("accepted") is True and compatibility.get("profile") == "h20_fla_reference_compat_v1",
            "The H20 FLA compatibility profile has not passed")
    require(compatibility.get("strict_regression_passed") is True, "Strict numerical regression did not pass")
    regression = Path((log_dir / "gdn_regression_path.txt").read_text().strip()).resolve()
    require(regression.is_relative_to(log_dir.resolve()), "Regression result path leaves this preflight")
    evidence = {"profile": HELPER_DIR / "h20_acceptance_profile.json", "harness": HELPER_DIR / "h20_gdn_regression.py",
                "plan": HELPER_DIR / "h20_gdn_regression_plan.json", "junit": log_dir / "upstream_gpu_suite.xml",
                "regression_report": regression / "report.json", "regression_checks": regression / "checks.jsonl"}
    for name, path in evidence.items():
        require(digest(path) == compatibility["evidence_sha256"][name], f"Compatibility evidence changed: {name}")
    helper_hashes = {name: digest(HELPER_DIR / name) for name in (
        "validate_h20_acceptance.py", "h20_acceptance.sh", "h20_smoke.sh", "h20_gdn_regression.py",
        "h20_acceptance_profile.json", "h20_gdn_regression_plan.json", "single_node_grpo_smoke.sh",
        "preflight_stage1.py", "preflight_run_v2.sh", "runtime_paths.sh", "h20_gdn_regression.sh",
        "restore_runtime.sh", "stage1_node.sh", "merlin_stage1_entry.sh")}
    grpo_path = Path((log_dir / "grpo_result_path.txt").read_text().strip()).resolve()
    require(grpo_path.is_relative_to((work / "runs").resolve()), "GRPO result path leaves this task's runs")
    grpo = json.loads(grpo_path.read_text())
    require(grpo.get("passed") is True and grpo.get("completed_training_steps") == 1,
            "No completed real GRPO step")
    require(grpo.get("trainer_gpus") == 4 and grpo.get("rollout_gpus") == 4 and grpo.get("executed_tool_calls", 0) > 0,
            "GRPO role allocation or actual interaction is unverified")
    require(math.isfinite(float(grpo["actor_grad_norm"])) and grpo["losses"]
            and all(math.isfinite(float(v)) for v in grpo["losses"].values()), "Invalid gradient/loss evidence")
    require(all(math.isfinite(float(grpo["update_timings"][k])) and float(grpo["update_timings"][k]) > 0
                for k in ("update_actor", "update_weights")), "Missing optimizer/weight sync")
    require(grpo["rollout_summary"]["finished"] >= 8 and grpo["rollout_summary"]["judged"] >= 4,
            "Insufficient real rollout groups")
    expected_checkpoints = {f"{kind}_world_size_4_rank_{rank}.pt"
                            for rank in range(4) for kind in ("model", "optim", "extra_state")}
    require(len(grpo["checkpoint_files"]) == 12 and {x["name"] for x in grpo["checkpoint_files"]} == expected_checkpoints,
            "Incomplete GRPO checkpoint inventory")
    for item in grpo["checkpoint_files"]:
        checkpoint = grpo_path.parent / "ckpt/global_step_1/actor" / item["name"]
        require(checkpoint.is_file() and checkpoint.stat().st_size == item["bytes"] > 0, "GRPO checkpoint changed")
    # Resolve only the venv directory: resolving bin/python would bypass the venv.
    py = Path(os.environ.get("VERL_ENV", str(work / "envs/verl"))).resolve() / "bin/python"
    require(py.is_file(), f"Prepared environment missing: {py}; do not install inside the 32-node job")
    runtime = Path("/opt/tiger/pear_runtime_20261114")
    bundle = json.loads((runtime / "runtime_bundle.json").read_text())
    require((runtime / ".restored_bundle").read_text().strip() == bundle["sha256"], "Local runtime restore marker differs from bundle")
    require(json.loads((root / "runtime_bundles/READY").read_text()) == bundle, "Published runtime differs from the local restored runtime")
    require(bundle["source_identity"]["source_sha256"] == source_hash, "Runtime was built from different source files")
    overlay = bundle.get("jit_overlay_provenance")
    require(overlay and digest(runtime / overlay["file"]) == overlay["sha256"], "JIT overlay provenance missing or changed")
    overlay_record = json.loads((runtime / overlay["file"]).read_text())
    require(overlay_record["effective_source_identity"] == bundle["source_identity"], "Overlay identity differs from runtime")
    model = Path(args.model_dir).resolve() if args.model_dir else work / "models/Qwen3.8-27B"
    record = model / ".pear_model"
    require(record.is_file(), "Official model download record missing")
    model_record = record.read_text()
    require("repository Qwen/Qwen3.8-27B" in model_record and "revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0" in model_record,
            "Base-model identity differs from approved recipe")
    for line in model_record.splitlines()[2:]:
        if line.strip():
            size, name = line.split(maxsplit=1)
            require((model / name).stat().st_size == int(size), f"Model file size differs: {name}")
    compat = Path(os.environ.get("PEAR_CUDA_COMPAT_DIR", str(root / "compat/cuda-compat-13-0_580.196.07/usr/local/cuda-13.0/compat"))).resolve()
    require((compat / "libcuda.so.1").is_file(), f"Compatibility library missing: {compat}")
    env = os.environ.copy()
    for name in ("CC", "CXX", "FC", "F77", "CMAKE_PREFIX_PATH", "CPATH", "LIBRARY_PATH", "PYTHONPATH", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "VIRTUAL_ENV"):
        env.pop(name, None)
    env["LD_LIBRARY_PATH"] = str(compat)
    inventory = command([str(py), "-c", "import importlib.metadata,json,platform; print(json.dumps([platform.python_version(),sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions())],sort_keys=True))"], env=env, cwd=repo)
    provenance = {
        "commit": COMMIT, "source_sha256": source_hash, "repo": str(repo), "work_dir": str(work),
        "model_dir": str(model), "model_record_sha256": digest(record),
        "environment_sha256": hashlib.sha256(inventory.encode()).hexdigest(),
        "compat_dir": str(compat), "libcuda_sha256": digest(compat / "libcuda.so.1"),
        "runtime_bundle": bundle,
        "preflight_marker_sha256": digest(passed), "preflight_logs": log_hashes,
        "acceptance_profile": compatibility, "validation_helpers": helper_hashes,
        "grpo_result_path": str(grpo_path), "grpo_result_sha256": digest(grpo_path),
        "candidate": {"gpu_family": "H20", "nodes": 32, "gpus_per_node": 8, "trainer_nodes": 8, "rollout_nodes": 24,
                      "lmcache_gb_per_tier": 600, "lmcache_tiers_per_rollout_node": 2, "ray_object_store_gib": 64},
    }
    hardware = check_hardware(compat, args.min_memory_gib)
    if args.certify:
        temporary = certificate.with_suffix(".json.partial")
        temporary.write_text(json.dumps({**provenance, "certified_at": time.time(), "preflight_hardware": hardware}, indent=2) + "\n")
        temporary.replace(certificate)
    else:
        require(certificate.is_file(), "Missing certificate; run --certify after dedicated eight-GPU preflight passes")
        saved = json.loads(certificate.read_text())
        require(all(saved.get(k) == v for k, v in provenance.items()), "Source/model/environment/preflight evidence differs from certificate")
    print(json.dumps({"status": "ready", "hostname": socket.gethostname(), **hardware, "certificate": str(certificate)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"PREFLIGHT FAILED: {exc}", file=sys.stderr)
        sys.exit(1)

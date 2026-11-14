#!/usr/bin/env python3
"""Fetch pinned public HF files with explicit proxy routing, resume, and full hashes."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(os.environ.get('PEAR_ROOT', '/mnt/bn/ic-vlm/personal/loongfei.li03/pear_20261114'))
REPO = 'Qwen/Qwen3.8-27B'
REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
MODEL = ROOT / 'work_dir/models/Qwen3.8-27B'
CACHE = MODEL / '.cache/pear_ranges'
PROXY = os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')
CHUNK = 32 * 1024 * 1024
PRINT_LOCK = threading.Lock()
DEADLINE = time.monotonic() + 12 * 3600

def log(event, **kw):
    with PRINT_LOCK:
        print(json.dumps({'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'event': event, **kw}), flush=True)

def opener(direct=False):
    return urllib.request.build_opener(urllib.request.ProxyHandler({} if direct or not PROXY else {'http': PROXY, 'https': PROXY}))

def digest(path, item):
    h = hashlib.sha256() if 'lfs' in item else hashlib.sha1()
    if 'lfs' not in item:
        h.update(f"blob {item['size']}\0".encode())
    with path.open('rb') as f:
        while data := f.read(8 * 1024 * 1024):
            h.update(data)
    return h.hexdigest()

def expected_hash(item):
    return item['lfs']['sha256'] if 'lfs' in item else item['blobId']

def resolve(item):
    name = item['rfilename']
    original = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{urllib.parse.quote(name, safe="/")}'
    with opener().open(urllib.request.Request(original, method='HEAD'), timeout=60) as response:
        url = response.url
    host = urllib.parse.urlsplit(url).hostname
    if host not in {'huggingface.co', 'us.aws.cdn.hf.co', 'cdn-lfs.huggingface.co', 'cdn-lfs-us-1.hf.co'}:
        raise RuntimeError(f'Unverified download hostname: {host}')
    return url, host != 'huggingface.co'

def fetch(item):
    name, size = item['rfilename'], int(item['size'])
    relative = Path(name)
    assert not relative.is_absolute() and '..' not in relative.parts
    dest = MODEL / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size == size and digest(dest, item) == expected_hash(item):
        log('verified_existing', file=name, bytes=size)
        return {'file': name, 'bytes': size, 'hash': expected_hash(item)}
    partial = CACHE / (name.replace('/', '__') + '.part')
    offset = partial.stat().st_size if partial.exists() else 0
    assert offset <= size, (name, offset, size)
    url = None
    failures = 0
    last_report = 0
    while offset < size:
        if time.monotonic() > DEADLINE:
            raise TimeoutError('Download exceeded the 12-hour retry window')
        try:
            if url is None:
                url, direct = resolve(item)
            end = min(offset + CHUNK, size) - 1
            req = urllib.request.Request(url, headers={'Range': f'bytes={offset}-{end}', 'Accept-Encoding': 'identity'})
            with opener(direct).open(req, timeout=60) as response:
                content_range = response.headers.get('Content-Range', '')
                if response.status == 206:
                    assert content_range.startswith(f'bytes {offset}-'), content_range
                    assert content_range.endswith(f'/{size}'), content_range
                else:
                    assert response.status == 200 and offset == 0 and size <= CHUNK, response.status
                remaining = end - offset + 1
                with partial.open('ab') as f:
                    while remaining:
                        data = response.read(min(256 * 1024, remaining))
                        if not data:
                            raise EOFError('Incomplete range')
                        f.write(data)
                        offset += len(data)
                        remaining -= len(data)
                    f.flush()
            failures = 0
            if time.monotonic() - last_report >= 30 or offset == size:
                log('progress', file=name, bytes=offset, total=size)
                last_report = time.monotonic()
        except Exception as e:
            # A valid downloaded prefix stays on disk. Never log signed URLs.
            offset = partial.stat().st_size if partial.exists() else 0
            failures += 1
            log('retry', file=name, bytes=offset, error=type(e).__name__, status=getattr(e, 'code', None), attempt=failures)
            url = None
            time.sleep(min(30, 2 ** min(failures, 5)))
    actual = digest(partial, item)
    if actual != expected_hash(item):
        raise RuntimeError(f'Hash mismatch for {name}; partial retained for diagnosis')
    partial.replace(dest)
    log('verified', file=name, bytes=size)
    return {'file': name, 'bytes': size, 'hash': actual}

def main():
    metadata = json.loads((ROOT / 'launch/model_metadata.json').read_text())
    assert metadata['id'] == REPO and metadata['sha'] == REVISION
    files = metadata['siblings']
    assert len(files) == 32 and sum(int(f['size']) for f in files) == 55586114863
    CACHE.mkdir(parents=True, exist_ok=True)
    log('start', repository=REPO, revision=REVISION, files=len(files), workers=4)
    results = []
    # Complete the small tokenizer/config files first, then bounded parallel shards.
    small = [f for f in files if int(f['size']) < CHUNK]
    large = [f for f in files if int(f['size']) >= CHUNK]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results.extend(pool.map(fetch, small))
        results.extend(pool.map(fetch, large))
    expected = {f['rfilename'] for f in files}
    actual = {str(p.relative_to(MODEL)) for p in MODEL.rglob('*') if p.is_file() and '.cache' not in p.relative_to(MODEL).parts and p.name != '.pear_model'}
    assert actual == expected, {'extra': sorted(actual - expected), 'missing': sorted(expected - actual)}
    lines = [f'repository {REPO}', f'revision {REVISION}']
    lines.extend(f"{f['size']} ./{f['rfilename']}" for f in sorted(files, key=lambda f: f['rfilename']))
    record = CACHE / 'pear_model.ready'
    record.write_text('\n'.join(lines) + '\n')
    record.replace(MODEL / '.pear_model')
    (ROOT / 'preflight/logs/model_integrity.json').write_text(json.dumps({'repository': REPO, 'revision': REVISION, 'files': results}, indent=2) + '\n')
    log('complete', bytes=sum(x['bytes'] for x in results), files=len(results))

if __name__ == '__main__':
    main()

"""Fetch and extract the main measurement traces into the current checkout."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import time
from http.client import IncompleteRead
from urllib.error import URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile

MANIFEST = 'https://raw.githubusercontent.com/LiCHOLjq/budgeted-evidence-reuse-data/main/manifest.json'
DOWNLOAD = 'https://github.com/LiCHOLjq/budgeted-evidence-reuse-data/releases/download/measurements/'
PART = re.compile(r'measurements-main-part[0-9]{2}\.zip')
# SHA-256 of the sorted LF-terminated member names of the original main dataset.
MEMBER_LIST_SHA256 = '04fde091f7bbcd0322af02ca816bcffaf279301f4a253ce265e984a14f2d9948'


def download(url, destination):
    request = Request(url, headers={'User-Agent': 'budgeted-evidence-reuse'})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=120) as response, destination.open('wb') as output:
                shutil.copyfileobj(response, output, 1024 * 1024)
            return
        except (URLError, IncompleteRead, OSError) as error:
            if attempt == 2:
                raise
            print(f'Download interrupted ({error}). Retrying...', flush=True)
            time.sleep(2 ** attempt)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fetch(asset, cache):
    name = asset['name']
    expected = asset['sha256']
    if not re.fullmatch(r'[a-f0-9]{64}', expected):
        raise ValueError(f'Invalid checksum for {name}')
    path = cache / name
    if path.is_file() and digest(path) == expected:
        return path
    print(f'Downloading {name}', flush=True)
    with tempfile.NamedTemporaryFile(dir=cache, suffix='.partial', delete=False) as file:
        temporary = Path(file.name)
    try:
        download(DOWNLOAD + name, temporary)
        if temporary.stat().st_size != asset['size'] or digest(temporary) != expected:
            raise ValueError(f'Download integrity check failed for {name}. Rerun to retry.')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def extract(parts, root):
    names = []
    for part in parts:
        with ZipFile(part) as archive:
            names.extend(archive.namelist())
    fingerprint = hashlib.sha256(('\n'.join(sorted(names)) + '\n').encode()).hexdigest()
    if fingerprint != MEMBER_LIST_SHA256:
        raise ValueError('The available ZIP parts do not cover the complete main dataset.')
    for part in parts:
        print(f'Extracting {part.name}', flush=True)
        with ZipFile(part) as archive:
            archive.extractall(root)


def main():
    root = Path.cwd()
    if not (root / 'experiments/benchmark_main.py').is_file():
        raise SystemExit('Run this command from the repository root in a fresh checkout.')
    cache = root / 'release-data/downloads'
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tempdir:
        metadata = Path(tempdir) / 'manifest.json'
        download(MANIFEST, metadata)
        release = json.loads(metadata.read_bytes())
    assets = sorted((a for a in release['assets'] if PART.fullmatch(a['name'])),
                    key=lambda a: a['name'])
    expected = [f'measurements-main-part{i:02d}.zip' for i in range(1, 14)]
    if [a['name'] for a in assets] != expected:
        raise SystemExit('Expected all 13 numbered main ZIP parts in the measurements release.')
    parts = [fetch(asset, cache) for asset in assets]
    extract(parts, root)
    print('Verified and extracted the complete main measurement collection.')


if __name__ == '__main__':
    main()


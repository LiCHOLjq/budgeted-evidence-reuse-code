"""Download, verify and normalize the upstream SNAP LiveJournal graph."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

root = Path(__file__).resolve().parents[1] / 'experiments'
data = root / 'data'
reference = data / 'livejournal_manifest.json'
record = json.loads(reference.read_bytes())
raw = data / 'soc-LiveJournal1.txt.gz'
if not raw.exists():
    partial = raw.with_suffix('.download')
    with urlopen(record['source_url'], timeout=120) as incoming, partial.open('wb') as target:
        shutil.copyfileobj(incoming, target)
    with partial.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != record['raw_sha256']:
            raise ValueError('Provider download differs from the recorded SNAP snapshot')
    partial.replace(raw)
with raw.open('rb') as stream:
    if hashlib.file_digest(stream, 'sha256').hexdigest() != record['raw_sha256']:
        raise ValueError('Local SNAP archive differs from the recorded snapshot')
manifest = data / 'livejournal/manifest.json'
if not manifest.exists():
    subprocess.run([sys.executable, str(root / 'prepare_livejournal.py'), '--data', str(data)], check=True)
rebuilt = json.loads(manifest.read_bytes())
for key in record:
    if key != 'elapsed_s' and rebuilt[key] != record[key]:
        raise ValueError(f'Metadata differ for {key}')
for name, info in record['files'].items():
    with (manifest.parent / name).open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != info['sha256']:
            raise ValueError(f'CSR file differs: {name}')
if manifest.read_bytes() != reference.read_bytes():
    shutil.copyfile(manifest, manifest.with_name('local_build.json'))
    shutil.copyfile(reference, manifest)
print('Verified LiveJournal input and both CSR relations.')

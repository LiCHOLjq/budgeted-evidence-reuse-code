"""Expand verified inputs used to reproduce the aggregate tables."""
import gzip
import hashlib
import json
from pathlib import Path

base = Path(__file__).resolve().parents[1] / 'experiments' / 'analysis_final'
expected = json.loads((base / 'analysis_audit.json').read_text())['output_sha256']
expected['strict_integrity.json'] = '48ed72ffe8be364329066e0a2a58232adf2c56ef1804a76f928668d28ca4c0a4'
for archive in sorted(base.glob('*.gz')):
    target = archive.with_suffix('')
    raw = gzip.decompress(archive.read_bytes())
    if hashlib.sha256(raw).hexdigest() != expected[target.name]:
        raise ValueError(f'Unexpected contents: {archive.name}')
    if target.exists():
        if target.read_bytes() != raw:
            raise FileExistsError(f'Existing file differs: {target.name}')
    else:
        target.write_bytes(raw)
    print(f'Verified {target.name}')

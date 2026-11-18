"""Normalize the official SNAP Wiki-Vote edge list without topology sampling.

Run: python artifact/prepare_wiki_vote_comparisons.py [--download]
The graph JSON contains the original directed edges; benchmark loaders may add
an explicitly named inverse relation. No vectors or workload are downloaded.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'comparisons'
URL = 'https://snap.stanford.edu/data/wiki-Vote.txt.gz'
PAGE = 'https://snap.stanford.edu/data/wiki-Vote.html'
EXPECTED_SHA256 = '7d3e53626e14b8b09fb3b396bece9d481ad606bd64ceab066349ff57d4ada7fc'


def prepare(download=False):
    DATA.mkdir(parents=True, exist_ok=True)
    raw_path = DATA / 'wiki-Vote.txt.gz'
    if download:
        raw = urlopen(URL, timeout=120).read()
        if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256:
            raise ValueError('Official bytes differ from the recorded comparisons snapshot; investigate before use.')
        raw_path.write_bytes(raw)
    raw = raw_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA256
    lines = gzip.decompress(raw).decode('utf8').splitlines()
    parsed = [tuple(map(int, line.split())) for line in lines
              if line.strip() and not line.startswith('#')]
    edges = sorted(set(parsed))
    original_ids = sorted({vertex for edge in edges for vertex in edge})
    mapping = {original: dense for dense, original in enumerate(original_ids)}
    assert (len(original_ids), len(parsed), len(edges)) == (7115, 103689, 103689)
    provenance = {
        'kind': 'Complete official SNAP Wiki-Vote directed topology; generated vectors/workloads are separate',
        'source_url': URL, 'source_page': PAGE, 'retrieved_date': '2026-10-25',
        'raw_sha256': EXPECTED_SHA256, 'raw_compressed_bytes': len(raw),
        'nodes': len(original_ids), 'directed_edges': len(edges),
        'duplicate_edges_removed': len(parsed)-len(edges),
        'self_loops': sum(src == dst for src, dst in edges),
        'direction': 'A -> B means user A voted on user B becoming an administrator',
        'snapshot_scope': 'Voting network through January 2008, as identified by SNAP',
        'transformation': 'Skip comment lines; parse integer pairs; sort distinct original IDs; map to dense 0-based IDs; sort directed edges. No component or edge subsampling.',
        'relation_convention': 'Stored edges are original vote direction only. Experimental relation 0 uses these edges; relation 1 is an explicitly generated inverse, if enabled.',
        'attribution': 'SNAP, Stanford University; dataset source papers by Jure Leskovec, Daniel Huttenlocher, and Jon Kleinberg (CHI 2010; WWW 2010); collection citation Jure Leskovec and Andrej Krevl (2014).',
        'license_statement': 'The inspected official dataset page gives source citations but no dataset-specific license declaration. No software or third-party dataset license is inferred.',
        'license_source_checked': PAGE,
    }
    payload = {'n': len(original_ids), 'original_ids': original_ids,
               'edges': [[mapping[src], mapping[dst]] for src, dst in edges],
               'provenance': provenance}
    target = DATA / 'wiki_vote.json'
    target.write_text(json.dumps(payload, separators=(',', ':'), ensure_ascii=False)+'\n', encoding='utf8')
    report = dict(provenance, normalized_json_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  normalized_file='artifact/data/comparisons/wiki_vote.json',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (DATA / 'wiki_vote_provenance.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    print(json.dumps({key: report[key] for key in ['nodes', 'directed_edges', 'raw_sha256',
                                                'normalized_json_sha256']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    prepare(parser.parse_args().download)

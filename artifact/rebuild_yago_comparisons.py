"""Reconstruct the paper’s YAGO topology from the upstream release or a checked CSV."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import deque
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
PROV = ROOT / 'comparisons' / 'provenance' / 'yago'
RAW_URL = 'https://yago-knowledge.org/data/yago4.5/yago-4.5.0.2-tiny.zip'
RAW_HASH = 'c9e52cf0338c9dd6586f4d601b02398fef55253a2be94dcace8ae9115f0d889d'
CSV_HASH = '1b2f265f880d4da69f58b43f8fce2cc8c63a2910e4982e01666039616f9e7338'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def rebuild_slice(csv_path, limit=2400):
    edges, undirected = [], {}
    total_rows = 0
    with csv_path.open(encoding='utf8', newline='') as handle:
        for row in csv.DictReader(handle):
            total_rows += 1
            if row['type'] != 'subClassOf':
                continue
            src, dst = int(row['src']), int(row['dst'])
            edges.append((src, dst))
            undirected.setdefault(src, set()).add(dst)
            undirected.setdefault(dst, set()).add(src)
    seed = max(undirected, key=lambda vertex: (len(undirected[vertex]), -vertex))
    queue, seen, order = deque([seed]), {seed}, []
    while queue and len(order) < limit:
        vertex = queue.popleft()
        order.append(vertex)
        for neighbor in sorted(undirected[vertex]):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    mapping = {vertex: index for index, vertex in enumerate(order)}
    selected = sorted({(mapping[src], mapping[dst]) for src, dst in edges
                       if src in mapping and dst in mapping})
    graph = {'n': len(order), 'original_ids': order,
             'edges': [list(edge) for edge in selected]}
    stats = {'csv_rows': total_rows, 'subClassOf_rows': len(edges),
             'bfs_start_original_id': seed, 'selected_nodes': len(order),
             'selected_directed_edges': len(selected),
             'explicit_forward_inverse_typed_edges': 2*len(selected),
             'all_edges_incident_to_dense_vertex_zero': all(0 in edge for edge in selected)}
    return graph, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=PROV/'edges.csv')
    parser.add_argument('--reference', type=Path,
                        default=None)
    parser.add_argument('--output', type=Path, default=ROOT/'experiments/data/yago_subclass_2400.json')
    parser.add_argument('--raw-zip', type=Path)
    parser.add_argument('--download-upstream', action='store_true')
    args = parser.parse_args()
    PROV.mkdir(parents=True, exist_ok=True)
    raw_path = args.raw_zip
    if args.download_upstream:
        raw_path = PROV/'upstream_cache/yago-4.5.0.2-tiny.zip'
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if not raw_path.exists():
            with urlopen(RAW_URL, timeout=120) as response, raw_path.open('wb') as handle:
                for block in iter(lambda: response.read(1024*1024), b''):
                    handle.write(block)
    upstream = None
    if raw_path is not None:
        actual = digest(raw_path)
        if actual != RAW_HASH:
            raise ValueError('Release checksum differs from the verified public comparisons release.')
        destination = PROV/'raw_rebuild'
        # The unchanged historical preprocessor uses locale-default output
        # encoding; force UTF-8 so Windows' GBK locale cannot truncate nodes.csv.
        command = [sys.executable, '-X', 'utf8', str(PROV/'preprocess_yago.py'), '--raw', str(raw_path),
                   '--out', str(destination), '--max-entities', '200000',
                   '--max-edges', '1000000', '--max-types-per-node', '4']
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        (PROV/'raw_preprocessing_stdout.txt').write_text(completed.stdout, encoding='utf8')
        rebuilt_hash = digest(destination/'edges.csv')
        if rebuilt_hash != CSV_HASH:
            raise AssertionError('Public release preprocessing did not reproduce shipped CSV bytes.')
        upstream = {'raw_url': RAW_URL, 'raw_sha256': actual,
                    'raw_bytes': raw_path.stat().st_size, 'csv_sha256': rebuilt_hash,
                    'csv_byte_identical': True,
                    'preprocessor_sha256': digest(PROV/'preprocess_yago.py'),
                    'parameters': {'max_entities': 200000, 'max_edges': 1000000,
                                   'max_types_per_node': 4, 'python_utf8_mode': True},
                    'stdout': completed.stdout.strip()}
        (PROV/'upstream_regeneration.json').write_text(json.dumps(upstream, indent=2)+'\n', encoding='utf8')
    if raw_path is not None and args.csv == PROV/'edges.csv':
        args.csv = destination/'edges.csv'
    actual_csv_hash = digest(args.csv)
    if actual_csv_hash != CSV_HASH:
        raise ValueError('CSV checksum differs from the source recorded in the retained ablation slice.')
    graph, stats = rebuild_slice(args.csv)
    checks = {}
    if args.reference is not None:
        reference = json.loads(args.reference.read_text(encoding='utf8'))
        checks = {key: graph[key] == reference[key] for key in ('n', 'original_ids', 'edges')}
        if not all(checks.values()):
            raise ValueError('Reconstructed topology differs from the supplied reference')
    encoded = json.dumps(graph, separators=(',', ':')).encode('utf8')
    if hashlib.sha256(encoded).hexdigest() != '90b5b4b4e29be6cbc716720dcaec0728f512c72cd7a375334a88c4900a3bb6d6':
        raise ValueError('Canonical YAGO graph differs from the measured topology')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded+b'\n')
    report = {
        'csv_sha256': actual_csv_hash,
        'reference_sha256': digest(args.reference) if args.reference else None,
        'canonical_graph_sha256': hashlib.sha256(encoded).hexdigest(),
        'rebuilt_file_sha256': digest(args.output), 'exact_graph_comparison': checks,
        'stats': stats, 'historical_metadata_note': 'Graph equality is checked independently. The historical absolute-path provenance strings are not copied into the rebuilt graph.',
        'source_url': RAW_URL, 'source_release_sha256': RAW_HASH,
        'raw_release_reprocessed_in_this_invocation': upstream is not None,
        'script_sha256': digest(Path(__file__)),
    }
    (PROV/'slice_regeneration.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

"""Regenerate the 647-row table from the reported or recomputed CSVs."""
import argparse
import csv
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', type=Path, default=root / 'tables',
                    help='Parent directory containing capacity/ and latency/')
parser.add_argument('--output', type=Path, default=root / 'recomputed/resources_tails_all.csv')
args = parser.parse_args()

def rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))

occupancy = {(r['config_id'], r['method'], r['phase']): r for r in
             rows(args.input / 'capacity/occupancy_eviction_configurations.csv')}
tails = rows(args.input / 'latency/service_query_tails_all_647.csv')
assert len(tails) == len({(r['config_id'], r['method']) for r in tails}) == 647
combined = []
for row in sorted(tails, key=lambda r: (r['config_id'], r['method'])):
    key = row['config_id'], row['method']
    full = occupancy[key + ('all',)]
    steady = occupancy[key + ('steady',)]
    assert int(steady['runs']) == int(row['runs'])
    combined.append(dict(config_id=key[0], method=key[1],
                         mean_occupancy=float(steady['mean_charged_fraction']),
                         peak_occupancy=float(full['maximum_observed_charged_fraction']),
                         eviction_mean_all=float(full['evicted_entries_mean_per_run']),
                         eviction_mean_steady=float(steady['evicted_entries_mean_per_run']),
                         query_p95_us=float(row['mean_query_p95_us']),
                         service_p95_us=float(row['mean_service_p95_us'])))
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open('w', encoding='utf8', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(combined[0]))
    writer.writeheader()
    writer.writerows(combined)

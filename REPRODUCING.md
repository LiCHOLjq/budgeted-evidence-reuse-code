# Reproducing the experiments

Start with [the 647 configuration/policy rows](tables/resources_tails_all.csv), [the configuration key](experiments/generated_assets/csv/configuration_ids.csv) or [the measurement archives](https://github.com/LiCHOLjq/budgeted-evidence-reuse-data/releases/tag/measurements). The tables report charged evidence memory and latency in the windows specified by their headers. Query latency and complete-service latency have separate columns. Process repeats share workloads, and the main LiveJournal experiment pairs each seed with one process.

## Reproduce the reported analyses

Use Python 3.11 or newer and NumPy. From a fresh checkout, run `python -m pip install -r requirements.txt`. The fetch command downloads all 13 main measurement ZIP parts from the [measurement release](https://github.com/LiCHOLjq/budgeted-evidence-reuse-data/releases/tag/measurements), checks each file against the SHA-256 digest in the data repository's manifest and extracts the complete set into the checkout. After an interrupted transfer, rerun the command to reuse verified parts.

```sh
python tools/fetch_main.py
python tools/unpack_analysis.py
python analyses/tails_ranking/analyze_tails_ranking.py --output recomputed/latency
python analyses/gate/analyze_gate_analysis.py --out recomputed/gate
python analyses/capacity_repeat/analyze_capacity_repeat.py --output recomputed/capacity
python tools/combine_tables.py --input recomputed
```

The analysis commands use the 5,999,680 accepted executions and untimed cache inventories to reproduce the published analyses of tail latency and ranking, adaptive routing, and occupancy and cross-family repetitions. Each analysis reads the relevant subset of the recorded runs. The scripts verify the SHA-256 hashes of the distributed measurement files. Measured source fingerprints in the run metadata describe the code used to collect those measurements. Paths and metadata checksums follow the public directory structure. The repository includes compressed intermediate tables and a public extract of the original analysis manifest, retaining the numerical output hashes. Compare regenerated CSVs with the matching files in `tables/latency`, `tables/gate` and `tables/capacity`. The capacity analysis also emits the full per-run occupancy CSV. The combined 647-row table joins these latency and occupancy results by configuration and method.

For the separate threshold follow-up, extract `measurements-threshold.zip` at the repository root and run `python experiments/threshold/analyze_threshold.py`. Outputs appear under `recomputed/threshold`. This analysis compares all threshold pairs across 172,800 executions and checks workload identities using per-seed hashes. The release also provides `measurements-historical.zip` for the earlier experiments. Each measurement collection includes a file inventory, checksums and coverage notes under `release-data/`. A fresh extraction preserves the expected paths.

## Run the executor

In a fresh checkout, the following runs semantic tests and benchmarks one synthetic workload. The six executor policies share the same queries, graph edits, vector updates and exact oracle witnesses. The output directory contains the recorded request traces, policy summaries and provenance for this new run.

```sh
python -m unittest discover -s artifact -p "test_engine_*.py"
python -m unittest discover -s experiments -p "test_engine_main.py"
python -m unittest discover -s experiments -p "test_runner_main.py"
python -m unittest discover -s analyses/gate -p "test_gate_analysis.py"
python experiments/benchmark_main.py --suite grid --seed 11 --repeat 0 --only-case 17cb31c686f09597 --output recomputed/synthetic
```

The [case registry](experiments/case_config.py), [execution protocol](experiments/EXPERIMENT_PROTOCOL_MAIN.md) and [evidence model](experiments/engine_protocol.md) describe the other configurations. The original timing campaign used Python 3.12.3, NumPy 2.5.4 and one pinned CPU with single-threaded BLAS. A new run on different hardware or library versions will have different timings and may make different adaptive decisions. Run new timing experiments in a checkout separate from extracted archive data.

## Retrieve graph inputs for new runs

All embeddings, queries and updates are generated. Obtain each topology from its original provider.

- **Wiki-Vote.** Run `python artifact/prepare_wiki_vote_comparisons.py --download`, then copy `artifact/data/comparisons/wiki_vote.json` to `experiments/data/wiki_vote.json`. The script verifies the original gzip and normalizes all 103,689 directed edges. Credit goes to J. Leskovec, D. Huttenlocher and J. Kleinberg and to the Stanford Network Analysis Project. [SNAP Wiki-Vote](https://snap.stanford.edu/data/wiki-Vote.html)
- **LiveJournal.** Run `python tools/prepare_livejournal.py` on a machine with sufficient memory and disk for the full input. The script verifies the original gzip, builds both CSR relations and checks the hashes of all four array files. The original graph has 68,993,773 directed edges. Credit goes to L. Backstrom, D. Huttenlocher, J. Kleinberg and X. Lan and to SNAP. [SNAP LiveJournal](https://snap.stanford.edu/data/soc-LiveJournal1.html)
- **YAGO.** Run `python artifact/rebuild_yago_comparisons.py --download-upstream` to retrieve YAGO 4.5.0.2 Tiny, check its SHA-256, preprocess it and build the 2,400-vertex subclass slice in `experiments/data/yago_subclass_2400.json`. The script verifies both the intermediate edge CSV and the canonical topology. The public JSON omits the original machine-specific provenance fields, so a new workload ID can differ even when its topology matches. YAGO credits Wikidata and Schema.org and publishes its data under CC BY-SA 3.0. [YAGO downloads and attribution](https://yago-knowledge.org/downloads/yago-4-5)

The provider pages describe the source data and applicable terms. The release archives contain the study's measurements. Original graph files and normalized edge lists can be reconstructed locally using the instructions above.


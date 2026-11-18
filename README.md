# Budgeted Evidence Reuse for Typed-Walk Filtering

This repository contains the query executor, benchmarks, and analysis scripts for the paper. Each query follows a specified sequence of edge types to find eligible vertices, then ranks those vertices using their current vectors. The study asks when a query can reuse earlier graph traversal work under a fixed memory budget as graph edges and vectors change independently.

The executor can cache a partial traversal result together with a record of the graph reads that produced it. For example, a vector update can change the ranking while leaving the eligible vertices unchanged. A graph edit to a recorded dependency, on the other hand, can require fresh traversal. The benchmarks measure how much work reuse saves after accounting for the cost of checking dependencies and evicting entries.

To explore the results, start with the [summary table](tables/resources_tails_all.csv). It reports cache occupancy, evictions, and 95th-percentile latency for each configuration and policy. Use the [configuration key](experiments/generated_assets/csv/configuration_ids.csv) to look up a row's workload. To run a small synthetic benchmark or regenerate the tables from the [experimental measurements](https://github.com/LiCHOLjq/budgeted-evidence-reuse-data), follow the [reproduction guide](REPRODUCING.md).

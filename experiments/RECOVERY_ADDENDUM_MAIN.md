# Diagnostic aggregation recovery, 2026-10-29

The frozen main campaign completed jobs 0–30. Job 31 (capacity diagnostic,
seed 11) stopped after query execution when the post-run summary attempted
to average physical/logical memory ratios containing `None`. The ratio is
undefined for an empty logical cache; it must not be replaced with zero.
All primary measurements and the balanced observer controls were completed
before this failure and remain unchanged.

`benchmark_recovery_main.py` imports the original frozen benchmark and replaces
exactly its post-query snapshot aggregation statement. Means use non-null
numeric samples; `sample_nonnull_count_python_to_logical_ratio` and
`sample_null_count_python_to_logical_ratio` explicitly retain both counts.
No query, maintenance, graph, ranking, diagnostic timer, workload, order,
executor, configuration, or random stream is changed. The original source
manifest remains present, and each recovered run/environment additionally
records `recovery_provenance`; a separate source snapshot binds this wrapper,
addendum and recovery manifest.

The incomplete `diagnostics/capacity_p0` is retained verbatim and excluded
from reported counts. Its canonical replacement is the fresh-process
`diagnostics/capacity_p0_recovery`, mapped to `diagnostics/capacity_p0` solely
for frozen-design coverage. All eight capacity cells are replayed, without
reusing timings from the failed attempt. The failed attempt's completed and
partial request traces are retained as operational failure evidence and are
not combined with the reported experimental ledger.

Jobs 32–39 retain their original schedule and output directories; they use
the same recovery wrapper so undefined ratios cannot interrupt later
diagnostic summaries. The original campaign/post status and logs remain
unchanged. `campaign_recovery_status.json` records the recovery schedule,
new PIDs, UTC times, exit codes and completion. A post-recovery dispatcher
then runs the already frozen threshold schedule and untimed gate inventory; no
timing processes overlap.

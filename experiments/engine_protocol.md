# main executor and diagnostics contract

This is new code under `experiments/`. It imports the frozen comparisons/replication modules from
`artifact/`; it does not edit or replace them. `EvidenceCache` is a factory with
the original positional API. Without `diagnostics=True`, existing methods return
their original comparisons/replication instances. The new method IDs are `EagerIndexLRU` and
`AdaptiveGate`. All policies use the original exact-length typed-walk semantics,
final anchor exclusion and current-vector exhaustive ranking.

## Mutation and reverse-index baseline

Call `apply_edge(cache, relation, src, dst, insert)` for **every** policy. It calls
`cache.g.edge` once and, for an effective edit, notifies a reverse-index executor.
It returns the effective-change Boolean. Vector replacement remains
`cache.g.vector(vertex, value)`. Apply complete batches between queries; do not
evaluate a query in the middle of a batch. Queries are serial.

`EagerIndexLRU` has the same fine, complete prefix certificates and forward
expansion/admission order as LocalLRU. It retains a map from each bucket ID to the
resident entry keys whose certificates contain that bucket. An effective graph
edit immediately removes all such entries, including entries dependent on an
empty read or an inherited read. Every removal unlinks all of that entry's
postings; empty index nodes are removed. Vector edits require no graph-evidence
invalidation. Lookup does not scan versions. Missing graph-update notifications
are detected by the graph epoch and conservatively flush the entire index/cache
at the next synchronization or notification, preserving exactness.

The logical persistent charge is:

- each entry: `72 + 4*path_length + 8*endpoint_count + 16*dependency_count`,
  plus `8*dependency_count` for reverse postings referencing the existing key;
- each nonempty reverse-index bucket node: 24 bytes (bucket ID, posting-list
  reference, count);
- a nonempty reverse-index root: 64 bytes.

`entry.size` includes the posting charge. `cache.index_bytes` includes node/root
charges, and `used = sum(entry.size) + index_bytes <= capacity` after every
mutation/admission. Admission may evict LRU entries while recomputing the number
of new shared index nodes; an entry still too large when alone is rejected. This
is not an optimal packing rule. Packed bucket IDs are `relation*n + source` and
assume a 64-bit representable key. The persistent charge describes packed fields,
not Python hash-table allocation or RSS. Per-policy control objects/statistical
counters are outside the original evidence accounting; additional gate selection
state is explicitly reserved below. Mutation-time invalidation and cleanup are
inside measured maintenance, not outside-timer diagnostics.

## Paid adaptive selector

`AdaptiveGate` uses LocalLRUMemo with persistent capacity `B-256` and EagerMemo
with no cross-request evidence. It reserves 256 logical bytes for fixed gate
configuration and scalar state (a 32-byte header/references, eight 8-byte
configuration fields and twenty 8-byte state slots). It requires `B >= 256`.
The declared defaults are a-priori: `gate_warmup=32`, `gate_period=256`,
`gate_block=8`, `gate_margin=.05`, `gate_alpha=.5`; pass `gate_seed` explicitly.
No parameter is fitted from held-out measurements.

Requests 1–32 execute **both** complete candidate queries, including both ranking
calls. There are then 256 ordinary requests. Requests 289–296 are the next paid
eight-probe block; later block starts are 545, 801, etc. Thus later starts are
spaced 256 requests apart, including the eight probes. Candidate execution order
alternates deterministically with `(gate_seed + zero_based_request) % 2`.

At each completed probe block, form each candidate's mean measured query time,
including its synchronization, then initialize/update the EWMA with weight .5.
Select the cache for **subsequent** requests only if its EWMA is at most .95 times
EagerMemo's EWMA. Initial service is EagerMemo. A probe returns the previously
selected candidate's result. Both candidate evaluations, timer reads, dictionary
bookkeeping, selection decisions and result handling remain inside the gate's
actual query timer. Shared graph/vector mutation is executed once externally and
included in actual complete service; the comparison does not duplicate that
common maintenance cost. The selected candidate is executed alone on nonprobes.

The dormant cache remains resident, consumes its full charge, and accumulates no
free shadow queries. Lazy version validation makes later reactivation exact.
Probe execution can warm it, but those builds and ranking work are paid. Every
candidate result is checked by the runner's oracle; the gate does not inspect a
future result or external measured policy label. This is a causal measured
selector, not an optimal policy, a certified predictor, or a guarantee of speed.
The initial cold pilot and retained stale entries may make it choose poorly.

`last_gate` reports request, probe flag, served policy, next selected policy,
candidate EWMA estimates, and measured candidate probe times when present.
`parameters` records the frozen values. `stats` adds gate probe/block/switch and
served-query counts, plus the sum of candidate elapsed probe time. The true gate
query timer additionally includes orchestration overhead. Original work counters
include **both** executed candidates on probes. The per-query memo proxy reports
the maximum of the executed candidates' memo peaks on a probe; sequential
candidate memo maps are released before the next candidate query. Ranked results,
endpoint output buffers and the separate persistent evidence remain outside this
memo-only proxy. It is not a combined process peak. A separate allocation
measurement is required for whole-system peak memory.

## Separate diagnostic replay

Construct existing/index policies with `diagnostics=True`; optionally also set
`profile=True`. The diagnostic get implementation preserves original state,
cache decisions and work counters but adds instrumentation. `last_diagnostics`
resets per query; `diagnostic_totals` accumulates:

`lookup_attempts`, `resident_probes`, `epoch_fast_path_hits`, `index_valid_hits`,
`validation_calls`, `validation_failures`, `validation_tokens`, `validation_ns`,
`final_hits`, `prefix_hits`, `query_misses`, `queries_after_graph_update`,
`graph_updates_since_previous_query`.

The epoch-fast-path numerator counts successful resident probes, not all requests.
Use per-request indicators to report the fraction of queries taking at least one
fast path, separately from lookup-level rates. Conditional checks and time must
filter requests with `queries_after_graph_update=1`; this means an **effective**
edit since the previous query, including any burst. A request with no immediate
edit can still need validation of an older entry. Query misses mean no reusable
prefix was found; lookup misses can include multiple failed probes per request.
Pure eager controls report zero cache-query misses because they make no probes.

`validation_ns` times only the version-comparison loop, including its iteration
and original work-counter operations. Failed-entry removal and post-validation
LRU handling are excluded. It is nested inside `lookup_validation_ns`; do not add
it again to the phase sum. Separate profile phases remain the original
`lookup_validation_ns`, `graph_ns`, `certificate_admission_ns`, `ranking_ns`,
`residual_ns`, `total_ns`. The fine-cache diagnostic traversal separates graph
expansion from dependency recording; it is not a causal decomposition of the
primary fused executor. Index admissions/unlinking in query time enter certificate
cost; eager invalidation performed by `apply_edge` enters maintenance instead.
Gate phase profiling is deliberately unsupported: its own paid probe ledger
already changes execution and must not be mixed with static diagnostic phases.

## Outside-timer inventories and workload matching

`overlap_audit(cache, changed_bucket_ids)` must be called **before** applying the
effective edits whose IDs it receives. The IDs are deduplicated. It returns
changed count, covered count/fraction, union-domain size, affected resident-entry
count and resident count. Stale but resident certificates are included. For
hybrid relation tokens, coverage is the conservative complete relation domain;
it is not the unavailable exact historical bucket read set. No cache state is
mutated. Capture each edit's pre-update overlap or one batch's pre-batch union,
and label those scopes rather than pooling them as the same quantity.

For hot-locality workloads, generate one trace using a declared reference cache
and frozen seed. Replay that identical trace for all compared policies. Selecting
updates separately from each measured policy's residency changes the workloads
and invalidates paired service comparisons. Report each measured policy's actual
overlap; the hot generator does not guarantee high overlap for every policy.

`cache_snapshot(cache, footprint=False)` scans resident state outside all primary
timers. It returns count, total logical used bytes, entry/index/gate charges,
certificate and endpoint totals, mean entry charge, mean total resident charge,
mean certificate count and cumulative lookup hits/misses. Entry mean excludes
shared index nodes; total resident mean allocates those nodes across entries.
Use separate replay/sampling controls rather than assuming these scans harmless.

With `footprint=True`, `resident_python_bytes` recursively sums `sys.getsizeof`
over entries, keys, endpoints, dependencies, and the reverse index, deduplicating
objects by identity across all roots. It excludes graph/vectors, active query
workspace, policy/counter objects and gate scalar state. Some primitive objects
may also be referenced by the graph/interpreter. The resulting ratio to the
corresponding logical resident charge is a **reachable-allocation estimate**,
not exclusive incremental memory, a packed implementation measurement or RSS.
It provides a CPython layout calibration with explicit scope. For an actual
incremental allocation measurement, run a separate tracemalloc baseline after
graph loading and before cache construction and report that observer's effect.

`hop_audit(graph, anchor, path)` is the original independent full-traversal
diagnostic. It returns every input/output frontier size, repeated-bucket and
exact/subset transfer work, terminal stability and final eligible count. Run it
after the batch and before/after the matching query without intervening edits.
Aggregate per-hop and eligible sizes by configuration with mean/median/p95;
eligibility fraction divides the final anchor-excluded count by graph.n.

## Correctness scope

The existing oracle independently traverses the graph but shares ranking code.
New tests also compare `rank` with `independent_rank`, a scalar Python distance
and lexicographic sort implementation, on exactly representable integer vectors,
ties, empty sets and k boundaries. It is not a proof of bitwise identical floating
reduction for arbitrary near-tie vectors. Do not retroactively describe the old
execution ledger as independently validating the numerical ranking kernel.

Tests check reverse-index bijection and total charge after bursts, no-op edits,
ABA updates, vector replacements, empty-read insertion, parent eviction, repeated
replacement and zero capacity. Diagnostic/profiled state is compared directly
with primary state. The gate tests verify paid probe scheduling, decision timing,
no decision before the initial block completes, dormant-cache validation and
fresh-oracle outputs.

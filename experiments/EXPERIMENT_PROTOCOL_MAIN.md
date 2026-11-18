# main prospective server experiment protocol

This protocol and `case_config_frozen.json` are frozen before reported measurements. Pilots are feasibility checks only and are excluded from every inferential table and execution ledger. Every case is retained, irrespective of direction or significance. The machine-readable case definitions and 40-job schedule are authoritative.

## Environment and isolation

The measurement server had an Intel Xeon w9-3475X, 72 logical CPUs, approximately 251 GiB RAM, and over 400 GiB available storage at preparation. Python 3.12.3 and NumPy 2.5.4 run in its dedicated virtual environment. The campaign supervisor and every child inherit CPU affinity 2; `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`. All jobs are serial, with no concurrent experiment or data-preparation job. Per-process PID, timestamps, affinity, versions, load, source/protocol hashes, and completion state are retained. Three sequential process invocations on one machine/session are not three independent deployments.

## Primary case families

All small-graph cases use seeds 11,29,47,61,73, three fresh processes, 512 requests per seed, 128 warmup requests and 384 subsequent steady-window requests. The primary methods are EagerMemo, LocalLRU, LocalLRUMemo, LocalHybrid, EagerIndexLRU, AdaptiveGate. Threshold families measure LocalHybrid only and include every threshold inside that same randomized family.

| Family | Cases per seed/process | Method executions |
|---|---:|---:|
| Wiki-Vote update probability 0,.05,.1,.3,.5,.8 × uniform/hot updates × h=3,4 |24|1,105,920|
| Wiki-Vote bursts of 16 edits every 32/160 requests × uniform/hot × h=3,4 |8|368,640|
| Wiki-Vote dimension 32,128,384,768 × h=3,4 |8|368,640|
| Wiki-Vote budget 64KiB,256KiB,1MiB,4MiB × h=3,4 |8|368,640|
| Synthetic degree 4,8,16,32 × length 3,4,5,6 × budget 64KiB/1MiB |32|1,474,560|
| Threshold 16,64,256,10^9 on all 32 synthetic cells and two Wiki-Vote cells |136|1,044,480|
| YAGO-star threshold 16,64,256,10^9 (diagnostic stress case) |4|30,720|
| Complete LiveJournal h=2, hot pool 8/800, three seeds |6 seed/cases|432,000|
| **Primary total** ||**5,193,600**|

Unless varied, graph update probability is .1, vector update probability .2, query hot-anchor probability .9, hot pool eight anchors, dimension32, Wiki budget1MiB, and hybrid threshold64. Synthetic graphs contain2,400vertices in disconnected200-vertex communities with two generated relations. Query paths have the declared exact length and shared alternating prefixes, with a final relation sampled from0/1. No query observes a partially applied update batch.

Large runs use complete SNAP LiveJournal with4,847,571vertices and68,993,773 original directed edges, plus an explicit inverse relation (137,987,546 typed edges). The official gzip, SHA256, normalized CSR files/counts/hashes, self-loop count and complete-selection statement are retained. No edges are sampled. All six methods use identical memory-mapped CSR plus copy-on-first-mutation bucket overlays; overlay lookup and mutation are included in service time. Vectors are synthetic32Dfloat64. Each seed11,29,47 has its own fresh process and both hot-pool cases,16MiBlogical capacity,12,000requests,2,000warmup and10,000steady-window requests. Seed and process are confounded here; do not infer separate process and seed variance. These are exact two-hop predicates, not long-path or ANN scalability evidence.

## Matching and event generation

Queries, graph-update draws, and vector-update draws use independent deterministic RNG streams. Coordinate sweeps use prefixes of the same768-dimensional normal stream for queries and small-graph initial/replacement vectors. Query anchors and paths therefore remain matched under rate, locality and dimensionality changes. Budget and threshold variants reuse the same workload when other factors agree. Within a suite/process, all case×seed tuples are globally shuffled; method order is shuffled per tuple. Baseline d32,1MiB,tau64 are rerun inside their respective sensitivity families, rather than borrowed from another suite.

Hot graph updates are generated once using an untimed, fixed LocalLRU reference cache at1MiB, regardless of tested method, budget, or threshold. Immediately before each generated edit, the hot bucket is drawn from the union of its resident fine certificate domains, including currently stale resident entries. If that union is empty the update falls back to uniform. A feasible insertion/deletion is selected; all replayed edits must be effective. The generated stream is frozen and replayed identically by all methods. Hot-reference targeting does not imply a hit on every evaluated policy's resident domain: separate diagnostics measure each policy's actual pre-batch overlap. Burst events are atomic relative to queries, start at the first positive multiple of the declared period, and contain16effective edge edits.

Cached workload identity includes normalized case factors, seed, generator-function SHA256, original input/CSR-manifest SHA256, and NumPy version. Loaded packs must match that identity. Packs record ordered query/update hashes and fresh oracle witnesses. Pilot caches cannot silently survive a generator/input/version change.

## Correctness, service and residency

The fresh oracle uses an independent set-comprehension traversal and the same float64 ranking kernel as the tested executor. It validates eligible-set cardinality/SHA256 and ordered top-k outputs; ranking arithmetic is not independently validated by this ledger. Separate scalar ranking tests in the engine suite provide limited independent coverage on exactly representable inputs. Every timed execution is checked, with zero tolerated endpoint, ordered-output, ineffective-update or capacity errors.

Service time begins before all graph/vector mutations and policy synchronization, and ends after query/ranking. Maintenance and query subtimes sum to service. Python allocation and GC remain enabled. Per-request primary bookkeeping outside service is O(1): resident count, used charge, entry charge and counters. No resident-certificate scan or Python-object traversal runs in primary measurement. Endpoint hashing/checking and output logging are outside service and common across methods.

`cache.used` includes all charged persistent metadata, including the eager reverse-index root,buckets and postings, or the adaptive gate's256bytes. Record charge is not Python RSS. Sparse inventory is exclusive to separate diagnostic replay; its exact accounting assertion is used=entry charge+index bytes+gate bytes. Every case retains per-request eligible count, per-hop frontier sizes, per-hop traversed edges, distinct read-bucket count and exact terminal-frontier stabilization index, permitting mean/median/p95 structural reporting without interpreting repeated edges as an optimality lower bound.

## New baselines and frozen adaptive decision

EagerIndexLRU maintains bucket→resident-entry postings; mutations use the same `apply_edge` wrapper for every method, with reverse invalidation charged to maintenance. Its logical capacity includes8bytes/posting,24bytes/nonempty indexed bucket and64bytes/nonempty root in addition to ordinary evidence. Missing notifications fail closed. Complete inherited and empty-read coverage is retained.

AdaptiveGate compares LocalLRUMemo against EagerMemo using paid paired probes on requests1–32, then8-request blocks beginning289,545,801,... . Both candidate traversals/rankings and the gate's decisions are paid in actual query time. Candidate order alternates deterministically with seed/request. The margin is.05 and EWMA alpha.5; initial selection is EagerMemo. Dormant persistent evidence remains resident and charged. The gate's memo-workspace proxy is the maximum of both executed candidates on a probe, excluding result buffers; it is not a process-peak measurement. All parameters are fixed independently of held-out outcomes.

For the synthetic grid, calibration/test cell assignment is the parity of degree-index+length-index+budget-index (even calibration, odd held out). All seeds/processes for a cell stay together. Calibration may fit only calibration-cell outcomes. Held-out labels must not select features, thresholds, gate parameters or stopping rules. Report prediction errors and paid policy regret, including negative results. A measured selection heuristic is not a competitive guarantee.

## Separate diagnostics and observer control

All24dynamic cases are replayed with all five seeds and the five static methods. All other primary families are diagnostically replayed with seed11, covering every case; the synthetic d32/h6/1MiB corner additionally uses the other four seeds. AdaptiveGate is excluded from profile instrumentation and instead supplies its paid per-request probe/decision ledger. The detailed replay includes pre-batch changed-bucket∩resident-domain overlap, epoch-fast-path outcomes, validation tokens/time conditional on graph-update requests, five query phases, every32nd resident inventory, and deduplicated reachable CPython `getsizeof` footprint. Hybrid relation tokens yield conservative expanded coverage. Resident footprint excludes graph,vectors,transient workspace and most policy bookkeeping; it is neither exclusive allocation nor RSS. One-seed calibrations do not receive population confidence intervals.

Instrumentation observer controls use the two Wiki winning cells and synthetic d32/h6/1MiB, five seeds,512requests and five static methods. Two balanced blocks execute separate plain/profile processes in opposite order. These contribute76,800plain control and76,800profile executions; they remain separate from primary outcomes. Full diagnostics, including these profile controls, total729,280executions. Combined primary, plain controls and diagnostics total5,999,680checked executions. These counts include repeated streams; distinct workload-hash×request coverage is reported separately.

Conditional seed bootstrap intervals resample paired seeds within a process/configuration. Process summaries retain each process result, mean,SD,range and clearly scoped descriptive intervals; do not pool repeated seeds as independent. Primary and diagnostic timing families are not mixed. Conditional validation_ns is nested within lookup_validation_ns and must not be added twice. The balanced plain/profile ratio quantifies the instrumented implementation's observer effect, rather than assuming zero overhead.

## Recovery and provenance

`run_campaign.py` launches40fresh child processes sequentially and retains `campaign_status.json`, PIDs, command arguments and logs. Completed case/method JSON files are never overwritten and may be skipped only if every current source hash matches their manifest. Every process archives the exact measured source and protocol copies. Incomplete work may be retried without deleting completed artifacts. No parameter is changed after inspecting reported outcomes; any failure requiring scientific-source modification ends this frozen campaign and requires a new explicitly labeled protocol/source version.

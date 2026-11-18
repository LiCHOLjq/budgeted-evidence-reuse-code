"""replication composition of persistent LocalLRU and request-local transfer memoization.

All pre-existing methods construct the frozen comparisons executor directly. Only
LocalLRUMemo uses this subclass. The graph is serial and immutable during a
request; both memo maps are discarded before ranking. Charges are logical,
not Python heap or RSS measurements.
"""
from time import perf_counter_ns
import engine_comparisons as comparisons

Graph, Entry = comparisons.Graph, comparisons.Entry
rank, oracle, hop_audit = comparisons.rank, comparisons.oracle, comparisons.hop_audit
METHODS = comparisons.METHODS + ['LocalLRUMemo']


class EvidenceCache(comparisons.EvidenceCache):
    def __new__(cls, graph, method, capacity, epoch_fast_path=True,
                profile=False, hybrid_tau=64):
        if method != 'LocalLRUMemo':
            # A base instance bypasses this subclass's __init__ and dispatch.
            return comparisons.EvidenceCache(graph, method, capacity, epoch_fast_path,
                                    profile, hybrid_tau)
        return super().__new__(cls)

    def __init__(self, graph, method, capacity, epoch_fast_path=True,
                 profile=False, hybrid_tau=64):
        super().__init__(graph, 'LocalLRU', capacity, epoch_fast_path,
                         profile, hybrid_tau)
        self.method = method

    @staticmethod
    def combined_memo_charge(input_frontier, output_frontier):
        # comparisons's 72-byte transfer header plus an 8-byte logical edge-cost scalar.
        return 80 + 8*len(input_frontier) + 8*len(output_frontier)

    def evidence(self, anchor, path):
        """Primary path: no phase clocks and no persistent memo map.

        deps never loses a dependency within this request. Every memo result
        was built after all of its supporting bucket versions were inserted
        into that same dictionary. Consequently exact and subset reuse need
        no second dependency scan. Snapshot certificates still copy deps in
        put(), independently of the lifetime of earlier cached prefixes.
        """
        self.last_memo_peak_bytes = self.last_memo_entries = 0
        e = self.get((anchor, tuple(path)))
        if e is not None:
            self.last_dependency_count = len(e.deps)
            return e.endpoints
        frontier, deps, cost, offset = frozenset({anchor}), {}, 0, 0
        for length in range(len(path)-1, 0, -1):
            e = self.get((anchor, tuple(path[:length])))
            if e is not None:
                frontier, deps, cost, offset = e.endpoints, dict(e.deps), e.edge_cost, length
                break
        memo, latest, workspace = {}, {}, 0
        for position in range(offset, len(path)):
            relation = path[position]
            key = (relation, frontier)
            record = memo.get(key)
            if record is not None:
                self.stats['memo_hits'] += 1
            else:
                self.stats['memo_misses'] += 1
                previous = latest.get(relation)
                if previous is not None and previous[0].issubset(frontier):
                    vertices = sorted(frontier-previous[0])
                    new, full_cost = set(previous[1][0]), previous[1][1]
                    self.stats['memo_subset_hits'] += 1
                    self.stats['memo_subset_saved_buckets'] += len(previous[0])
                else:
                    vertices, new, full_cost = sorted(frontier), set(), 0
                self.stats['bucket_expansions'] += len(vertices)
                for vertex in vertices:
                    bucket = relation*self.g.n+vertex
                    deps[bucket] = self.g.versions[bucket]  # Empty reads count.
                    neighbors = self.g.adj[relation][vertex]
                    new.update(neighbors)
                    full_cost += len(neighbors)
                    self.stats['edge_visits'] += len(neighbors)
                record = frozenset(new), full_cost
                memo[key] = record
                workspace += self.combined_memo_charge(frontier, record[0])
                if previous is None:
                    workspace += 24  # Relation ID, input reference, record reference.
            latest[relation] = (frontier, record)
            frontier = record[0]
            # Preserve ordinary LocalLRU's logical cold-prefix edge metadata;
            # actual expansion work is reported separately in stats.
            cost += record[1]
            self.put((anchor, tuple(path[:position+1])), frontier, deps, cost)
        self.last_memo_peak_bytes = workspace
        self.last_memo_entries = len(memo)
        self.last_dependency_count = len(deps)
        return frontier

    def _profiled_evidence(self, anchor, path, phases):
        """Diagnostic path separates expansion and dependency-recording loops.

        Timers and the second loop perturb execution. These phases describe
        this diagnostic implementation, not an exact decomposition of primary
        timings. Memo lookup/subset construction is lookup; result copying is
        graph; memo storage and certificate admission share certificate time.
        """
        self.last_memo_peak_bytes = self.last_memo_entries = 0
        start = perf_counter_ns()
        e = self.get((anchor, tuple(path)))
        phases['lookup_validation_ns'] += perf_counter_ns()-start
        if e is not None:
            self.last_dependency_count = len(e.deps)
            return e.endpoints
        start = perf_counter_ns()
        frontier, deps, cost, offset = frozenset({anchor}), {}, 0, 0
        phases['certificate_admission_ns'] += perf_counter_ns()-start
        for length in range(len(path)-1, 0, -1):
            start = perf_counter_ns()
            e = self.get((anchor, tuple(path[:length])))
            phases['lookup_validation_ns'] += perf_counter_ns()-start
            if e is not None:
                start = perf_counter_ns()
                frontier, deps, cost, offset = e.endpoints, dict(e.deps), e.edge_cost, length
                phases['certificate_admission_ns'] += perf_counter_ns()-start
                break
        start = perf_counter_ns()
        memo, latest, workspace = {}, {}, 0
        phases['certificate_admission_ns'] += perf_counter_ns()-start
        for position in range(offset, len(path)):
            start = perf_counter_ns()
            relation = path[position]
            key = (relation, frontier)
            record = memo.get(key)
            if record is not None:
                self.stats['memo_hits'] += 1
            else:
                self.stats['memo_misses'] += 1
                previous = latest.get(relation)
                use_subset = previous is not None and previous[0].issubset(frontier)
                if use_subset:
                    vertices = sorted(frontier-previous[0])
                    self.stats['memo_subset_hits'] += 1
                    self.stats['memo_subset_saved_buckets'] += len(previous[0])
                else:
                    vertices = sorted(frontier)
            phases['lookup_validation_ns'] += perf_counter_ns()-start
            if record is None:
                start = perf_counter_ns()
                new = set(previous[1][0]) if use_subset else set()
                full_cost = previous[1][1] if use_subset else 0
                self.stats['bucket_expansions'] += len(vertices)
                for vertex in vertices:
                    neighbors = self.g.adj[relation][vertex]
                    new.update(neighbors)
                    full_cost += len(neighbors)
                    self.stats['edge_visits'] += len(neighbors)
                record = frozenset(new), full_cost
                phases['graph_ns'] += perf_counter_ns()-start
                start = perf_counter_ns()
                for vertex in vertices:
                    bucket = relation*self.g.n+vertex
                    deps[bucket] = self.g.versions[bucket]
                memo[key] = record
                workspace += self.combined_memo_charge(frontier, record[0])
                if previous is None:
                    workspace += 24
                phases['certificate_admission_ns'] += perf_counter_ns()-start
            start = perf_counter_ns()
            latest[relation] = (frontier, record)
            frontier = record[0]
            cost += record[1]
            self.put((anchor, tuple(path[:position+1])), frontier, deps, cost)
            phases['certificate_admission_ns'] += perf_counter_ns()-start
        self.last_memo_peak_bytes = workspace
        self.last_memo_entries = len(memo)
        self.last_dependency_count = len(deps)
        return frontier

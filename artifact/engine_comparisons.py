"""Serial exact walk executor, comparison configuration.

Capacity is a logical packed-record charge, never Python RSS. Negative dependency
keys encode relation epochs; nonnegative keys encode adjacency buckets. Profiling
uses a separate execution path so headline timings contain no per-phase clocks.
"""
from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter_ns
import numpy as np

METHODS = ['Eager', 'EagerSorted', 'EagerMemo', 'BothEpochLRU',
           'GraphEpochLRU', 'LocalLRU', 'LocalDensity', 'LocalFinal', 'LocalHybrid']


class Graph:
    def __init__(self, adjacency, vectors):
        self.adj = [[set(x) for x in relation] for relation in adjacency]
        self.n = len(vectors)
        self.versions = [0] * (len(adjacency) * self.n)
        self.relation_epochs = [0] * len(adjacency)
        self.vectors = np.asarray(vectors, dtype=np.float64).copy()
        self.graph_epoch = self.vector_epoch = 0

    def edge(self, relation, src, dst, insert):
        bucket = self.adj[relation][src]
        if (dst in bucket) == insert:
            return False
        (bucket.add if insert else bucket.remove)(dst)
        self.versions[relation*self.n+src] += 1
        self.relation_epochs[relation] += 1
        self.graph_epoch += 1
        return True

    def vector(self, vertex, value):
        self.vectors[vertex] = value
        self.vector_epoch += 1


@dataclass
class Entry:
    endpoints: frozenset
    deps: dict
    edge_cost: int
    size: int
    frequency: int = 1
    touched: int = 0
    validated_epoch: int = 0


class EvidenceCache:
    def __init__(self, graph, method, capacity, epoch_fast_path=True,
                 profile=False, hybrid_tau=64):
        if method not in METHODS:
            raise ValueError('Unknown method: '+str(method))
        if capacity < 0 or hybrid_tau < 0:
            raise ValueError('capacity and hybrid_tau must be nonnegative')
        self.g, self.method, self.capacity = graph, method, capacity
        self.epoch_fast_path = epoch_fast_path
        self.profile, self.hybrid_tau = profile, hybrid_tau
        self.entries = OrderedDict()
        self.used = self.peak = self.clock = 0
        self.last_graph = graph.graph_epoch
        self.last_vector = graph.vector_epoch
        self.last_dependency_count = self.last_endpoint_count = 0
        self.last_memo_peak_bytes = self.last_memo_entries = 0
        self.last_profile_ns = {}
        self.stats = dict(hits=0, misses=0, invalidated=0, evicted=0,
                          rejected=0, validated_buckets=0, edge_visits=0, flushes=0,
                          bucket_expansions=0, memo_hits=0, memo_misses=0,
                          memo_subset_hits=0, memo_subset_saved_buckets=0)

    @staticmethod
    def charge(key, endpoints, deps):
        # Logical charge only: either kind of dependency costs 16 bytes.
        return 72 + 4*len(key[1]) + 8*len(endpoints) + 16*len(deps)

    def drop(self, key, reason):
        self.used -= self.entries.pop(key).size
        self.stats[reason] += 1

    def synchronize(self):
        flush = (self.g.graph_epoch != self.last_graph and
                 self.method in ('GraphEpochLRU', 'BothEpochLRU'))
        flush |= self.g.vector_epoch != self.last_vector and self.method == 'BothEpochLRU'
        if flush:
            self.stats['invalidated'] += len(self.entries)
            self.entries.clear()
            self.used = 0
            self.stats['flushes'] += 1
        self.last_graph, self.last_vector = self.g.graph_epoch, self.g.vector_epoch

    def get(self, key):
        e = self.entries.get(key)
        if e is None:
            self.stats['misses'] += 1
            return None
        if self.method.startswith('Local') and (
                not self.epoch_fast_path or e.validated_epoch != self.g.graph_epoch):
            for bucket, version in e.deps.items():
                self.stats['validated_buckets'] += 1
                current = (self.g.versions[bucket] if bucket >= 0 else
                           self.g.relation_epochs[-1-bucket])
                if current != version:
                    self.drop(key, 'invalidated')
                    self.stats['misses'] += 1
                    return None
            e.validated_epoch = self.g.graph_epoch
        self.clock += 1
        e.touched = self.clock
        e.frequency += 1
        self.entries.move_to_end(key)
        self.stats['hits'] += 1
        return e

    def put(self, key, endpoints, deps, cost):
        size = self.charge(key, endpoints, deps)
        if size > self.capacity:
            self.stats['rejected'] += 1
            return
        if key in self.entries:
            self.used -= self.entries.pop(key).size
        while self.used+size > self.capacity and self.entries:
            if self.method == 'LocalDensity':
                victim = min(self.entries, key=lambda k: (
                    self.entries[k].frequency*max(1, self.entries[k].edge_cost)/
                    self.entries[k].size, self.entries[k].touched))
            else:
                victim = next(iter(self.entries))
            self.drop(victim, 'evicted')
        self.clock += 1
        self.entries[key] = Entry(frozenset(endpoints), dict(deps), cost, size,
                                 touched=self.clock, validated_epoch=self.g.graph_epoch)
        self.used += size
        self.peak = max(self.peak, self.used)
        assert self.used <= self.capacity

    def _record_hybrid(self, deps, relation, vertices):
        """Compress a relation once its distinct cumulative bucket count > tau.

        deps was freshly built or inherited ONLY from get() after validation.
        Serial execution ensures all recorded versions are still current here.
        A coarse tag inherited from a hit must never be refreshed opportunistically.
        """
        token = -1-relation
        if token in deps or not vertices:
            return
        lower, upper = relation*self.g.n, (relation+1)*self.g.n
        fine_keys = [key for key in deps if lower <= key < upper]
        count = len(fine_keys)
        for vertex in vertices:
            bucket = lower+vertex
            if bucket not in deps:
                count += 1
                if count > self.hybrid_tau:
                    # Includes inherited and newly recorded fine dependencies.
                    for key in fine_keys:
                        del deps[key]
                    deps[token] = self.g.relation_epochs[relation]
                    return
                fine_keys.append(bucket)
            deps[bucket] = self.g.versions[bucket]  # Empty reads count too.

    def evidence(self, anchor, path):
        """Uninstrumented primary path. The working dictionary mutates in place."""
        if self.method == 'EagerMemo':
            return self._memo_evidence(anchor, path)
        if self.method in ('Eager', 'EagerSorted'):
            frontier = {anchor}
            for relation in path:
                new = set()
                vertices = sorted(frontier) if self.method == 'EagerSorted' else frontier
                self.stats['bucket_expansions'] += len(frontier)
                for vertex in vertices:
                    bucket = self.g.adj[relation][vertex]
                    self.stats['edge_visits'] += len(bucket)
                    new.update(bucket)
                frontier = new
            self.last_dependency_count = 0
            return frozenset(frontier)
        e = self.get((anchor, tuple(path)))
        if e is not None:
            self.last_dependency_count = len(e.deps)
            return e.endpoints
        frontier, deps, cost, start = {anchor}, {}, 0, 0
        if self.method != 'LocalFinal':
            for length in range(len(path)-1, 0, -1):
                e = self.get((anchor, tuple(path[:length])))
                if e is not None:
                    frontier, deps, cost, start = e.endpoints, dict(e.deps), e.edge_cost, length
                    break
        for position in range(start, len(path)):
            relation, new = path[position], set()
            vertices = sorted(frontier)
            self.stats['bucket_expansions'] += len(vertices)
            if self.method == 'LocalHybrid':
                self._record_hybrid(deps, relation, vertices)
                for vertex in vertices:
                    neighbors = self.g.adj[relation][vertex]
                    new.update(neighbors)
                    cost += len(neighbors)
                    self.stats['edge_visits'] += len(neighbors)
            else:
                for vertex in vertices:
                    bucket_id = relation*self.g.n+vertex
                    deps[bucket_id] = self.g.versions[bucket_id]
                    neighbors = self.g.adj[relation][vertex]
                    new.update(neighbors)
                    cost += len(neighbors)
                    self.stats['edge_visits'] += len(neighbors)
            frontier = frozenset(new)
            if self.method != 'LocalFinal' or position == len(path)-1:
                self.put((anchor, tuple(path[:position+1])), frontier, deps, cost)
        self.last_dependency_count = len(deps)
        return frozenset(frontier)

    @staticmethod
    def memo_charge(input_frontier, output_frontier):
        # Declared packed logical workspace, not Python heap/RSS. Each entry
        # charges 64 header bytes, 8 relation bytes, and both uint64 ID arrays.
        # Shared frozensets are deliberately charged per entry, conservatively.
        return 72 + 8*len(input_frontier) + 8*len(output_frontier)

    def _memo_evidence(self, anchor, path):
        """Exact and subset-delta per-request memo, discarded on return.

        (r,F) is sufficient only under the serial, immutable-during-query
        graph contract. For a previous transfer (r,A)->T_r(A), A subset F
        permits T_r(F)=T_r(A) union T_r(F-A). No equality of one hop
        authorizes skipping later relations.
        """
        memo, latest, frontier, workspace = {}, {}, frozenset({anchor}), 0
        self.last_memo_peak_bytes = self.last_memo_entries = 0
        for relation in path:
            key = (relation, frontier)
            result = memo.get(key)
            if result is not None:
                self.stats['memo_hits'] += 1
                latest[relation] = (frontier, result)
                frontier = result
                continue
            self.stats['memo_misses'] += 1
            previous = latest.get(relation)
            if previous is not None and previous[0].issubset(frontier):
                vertices = frontier-previous[0]
                new = set(previous[1])
                self.stats['memo_subset_hits'] += 1
                self.stats['memo_subset_saved_buckets'] += len(previous[0])
            else:
                vertices, new = frontier, set()
            self.stats['bucket_expansions'] += len(vertices)
            for vertex in vertices:
                neighbors = self.g.adj[relation][vertex]
                self.stats['edge_visits'] += len(neighbors)
                new.update(neighbors)
            result = frozenset(new)
            memo[key] = result
            workspace += self.memo_charge(frontier, result)
            if previous is None:
                workspace += 24  # Relation ID + two references, logical only.
            latest[relation] = (frontier, result)
            frontier = result
        self.last_memo_peak_bytes = workspace  # No entries are evicted within a request.
        self.last_memo_entries = len(memo)
        self.last_dependency_count = 0
        return frontier

    def _profiled_memo_evidence(self, anchor, path, phases):
        start = perf_counter_ns()
        memo, latest, frontier, workspace = {}, {}, frozenset({anchor}), 0
        self.last_memo_peak_bytes = self.last_memo_entries = 0
        phases['certificate_admission_ns'] += perf_counter_ns()-start
        for relation in path:
            start = perf_counter_ns()
            key = (relation, frontier)
            result = memo.get(key)
            if result is not None:
                self.stats['memo_hits'] += 1
                previous, vertices, use_subset = None, None, False
            else:
                self.stats['memo_misses'] += 1
                previous = latest.get(relation)
                use_subset = previous is not None and previous[0].issubset(frontier)
                if use_subset:
                    vertices = frontier-previous[0]
                    self.stats['memo_subset_hits'] += 1
                    self.stats['memo_subset_saved_buckets'] += len(previous[0])
                else:
                    vertices = frontier
            phases['lookup_validation_ns'] += perf_counter_ns()-start
            if result is not None:
                start = perf_counter_ns()
                latest[relation] = (frontier, result)
                phases['certificate_admission_ns'] += perf_counter_ns()-start
                frontier = result
                continue
            start = perf_counter_ns()
            self.stats['bucket_expansions'] += len(vertices)
            new = set(previous[1]) if use_subset else set()
            for vertex in vertices:
                neighbors = self.g.adj[relation][vertex]
                self.stats['edge_visits'] += len(neighbors)
                new.update(neighbors)
            result = frozenset(new)
            phases['graph_ns'] += perf_counter_ns()-start
            start = perf_counter_ns()
            memo[key] = result
            workspace += self.memo_charge(frontier, result)
            if previous is None:
                workspace += 24
            latest[relation] = (frontier, result)
            phases['certificate_admission_ns'] += perf_counter_ns()-start
            frontier = result
        self.last_memo_peak_bytes = workspace
        self.last_memo_entries = len(memo)
        self.last_dependency_count = 0
        return frontier

    def _profiled_evidence(self, anchor, path, phases):
        if self.method == 'EagerMemo':
            return self._profiled_memo_evidence(anchor, path, phases)
        if self.method in ('Eager', 'EagerSorted'):
            start = perf_counter_ns()
            result = self.evidence(anchor, path)
            phases['graph_ns'] += perf_counter_ns()-start
            return result
        start = perf_counter_ns()
        e = self.get((anchor, tuple(path)))
        phases['lookup_validation_ns'] += perf_counter_ns()-start
        if e is not None:
            self.last_dependency_count = len(e.deps)
            return e.endpoints
        frontier, deps, cost, offset = {anchor}, {}, 0, 0
        if self.method != 'LocalFinal':
            for length in range(len(path)-1, 0, -1):
                start = perf_counter_ns()
                e = self.get((anchor, tuple(path[:length])))
                phases['lookup_validation_ns'] += perf_counter_ns()-start
                if e is not None:
                    start = perf_counter_ns()
                    frontier, deps, cost, offset = e.endpoints, dict(e.deps), e.edge_cost, length
                    phases['certificate_admission_ns'] += perf_counter_ns()-start
                    break
        for position in range(offset, len(path)):
            start = perf_counter_ns()
            relation, new = path[position], set()
            vertices = sorted(frontier)
            self.stats['bucket_expansions'] += len(vertices)
            for vertex in vertices:
                neighbors = self.g.adj[relation][vertex]
                new.update(neighbors)
                cost += len(neighbors)
                self.stats['edge_visits'] += len(neighbors)
            frontier = frozenset(new)
            phases['graph_ns'] += perf_counter_ns()-start
            start = perf_counter_ns()
            if self.method == 'LocalHybrid':
                self._record_hybrid(deps, relation, vertices)
            else:
                for vertex in vertices:
                    bucket = relation*self.g.n+vertex
                    deps[bucket] = self.g.versions[bucket]
            if self.method != 'LocalFinal' or position == len(path)-1:
                self.put((anchor, tuple(path[:position+1])), frontier, deps, cost)
            phases['certificate_admission_ns'] += perf_counter_ns()-start
        self.last_dependency_count = len(deps)
        return frozenset(frontier)

    def query(self, anchor, path, q, k=10):
        if k < 0:
            raise ValueError('k must be nonnegative')
        if self.profile:
            return self._profiled_query(anchor, path, q, k)
        self.synchronize()
        endpoints = self.evidence(anchor, path)-{anchor}
        self.last_endpoint_count = len(endpoints)
        return rank(self.g.vectors, endpoints, q, k), endpoints

    def _profiled_query(self, anchor, path, q, k):
        total_start = perf_counter_ns()
        phases = dict(lookup_validation_ns=0, graph_ns=0,
                      certificate_admission_ns=0, ranking_ns=0)
        start = perf_counter_ns()
        self.synchronize()
        phases['lookup_validation_ns'] += perf_counter_ns()-start
        endpoints = self._profiled_evidence(anchor, path, phases)
        start = perf_counter_ns()
        endpoints = endpoints-{anchor}
        self.last_endpoint_count = len(endpoints)
        result = rank(self.g.vectors, endpoints, q, k), endpoints
        phases['ranking_ns'] += perf_counter_ns()-start
        total = perf_counter_ns()-total_start
        phases['residual_ns'] = total-sum(phases.values())
        phases['total_ns'] = total
        self.last_profile_ns = phases
        return result


def rank(vectors, endpoints, q, k):
    if k < 0:
        raise ValueError('k must be nonnegative')
    if not endpoints:
        return ()
    ids = np.fromiter(sorted(endpoints), dtype=np.int64)
    delta = vectors[ids]-q
    distances = np.sum(delta*delta, axis=1)
    order = np.lexsort((ids, distances))[:k]
    return tuple(int(ids[i]) for i in order)


def oracle(graph, anchor, path, q, k=10):
    current = [anchor]
    for relation in path:
        seen = set()
        for source in current:
            for target in graph.adj[relation][source]:
                seen.add(target)
        current = list(seen)
    endpoints = frozenset(x for x in current if x != anchor)
    return rank(graph.vectors, endpoints, q, k), endpoints


def hop_audit(graph, anchor, path):
    """Independent untimed full expansion audit; never called by query().

    A stable suffix is retrospective equality of all remaining full frontiers,
    NOT a proof permitting early termination for an arbitrary relation suffix.
    Counts include empty bucket reads. Returned rows contain scalar diagnostics,
    not retained frontier objects, and can be serialized directly to JSON/CSV.
    """
    current = frozenset({anchor})
    frontiers, seen_transfers, seen_buckets, rows, latest = [current], {}, set(), [], {}
    for hop, relation in enumerate(path, 1):
        key = (relation, current)
        first_hop = seen_transfers.get(key)
        seen_transfers.setdefault(key, hop)
        previous = latest.get(relation)
        subset = first_hop is None and previous is not None and previous.issubset(current)
        memo_vertices = (frozenset() if first_hop is not None else
                         current-previous if subset else current)
        edges, new_buckets, new, memo_edges = 0, 0, set(), 0
        for vertex in current:
            bucket_key = (relation, vertex)
            if bucket_key not in seen_buckets:
                new_buckets += 1
                seen_buckets.add(bucket_key)
            neighbors = graph.adj[relation][vertex]
            edges += len(neighbors)
            if vertex in memo_vertices:
                memo_edges += len(neighbors)
            new.update(neighbors)
        output = frozenset(new)
        rows.append(dict(hop=hop, relation=relation, input_frontier_size=len(current),
                         output_frontier_size=len(output), edge_visits=edges,
                         bucket_expansions=len(current), new_distinct_buckets=new_buckets,
                         repeated_bucket_expansions=len(current)-new_buckets,
                         frontier_unchanged=int(output == current),
                         cardinality_nongrowing=int(len(output) <= len(current)),
                         repeated_transfer=int(first_hop is not None),
                         subset_transfer=int(subset), memo_bucket_expansions=len(memo_vertices),
                         memo_edge_visits=memo_edges,
                         transfer_first_hop=hop if first_hop is None else first_hop))
        latest[relation] = current
        current = output
        frontiers.append(current)
    stable = next((i for i in range(len(path))
                   if all(frontiers[j] == frontiers[i] for j in range(i+1, len(frontiers)))), None)
    total_edges = sum(row['edge_visits'] for row in rows)
    repeated_edges = sum(row['edge_visits'] for row in rows if row['repeated_transfer'])
    subset_edges = sum(row['edge_visits']-row['memo_edge_visits'] for row in rows if row['subset_transfer'])
    after_stable = sum(row['edge_visits'] for row in rows if stable is not None and row['hop'] > stable)
    return dict(hops=rows, total_edge_visits=total_edges,
                total_bucket_expansions=sum(row['bucket_expansions'] for row in rows),
                distinct_buckets=len(seen_buckets), distinct_transfers=len(seen_transfers),
                repeated_transfer_steps=sum(row['repeated_transfer'] for row in rows),
                edge_visits_at_repeated_transfers=repeated_edges,
                repeated_transfer_edge_fraction=repeated_edges/total_edges if total_edges else 0.0,
                subset_transfer_steps=sum(row['subset_transfer'] for row in rows),
                edge_visits_avoided_by_subset=subset_edges,
                memo_executed_edge_visits=sum(row['memo_edge_visits'] for row in rows),
                memo_executed_bucket_expansions=sum(row['memo_bucket_expansions'] for row in rows),
                memo_edge_savings_fraction=(repeated_edges+subset_edges)/total_edges if total_edges else 0.0,
                terminal_stable_frontier_start_hop=stable,
                post_stable_frontier_edge_visits=after_stable,
                post_stable_frontier_edge_fraction=after_stable/total_edges if total_edges else 0.0,
                final_endpoint_count=len(current-{anchor}))

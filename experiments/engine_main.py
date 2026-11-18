"""main controls and opt-in diagnostics; frozen comparisons/replication executors are imported.

Use apply_edge(cache, ...) for every effective graph mutation. The reverse-index
control fails closed on missing notifications. Diagnostic scans, object-size
inventory and hop_audit must be run in a separate diagnostic replay.
"""
from collections import OrderedDict
from pathlib import Path
from time import perf_counter_ns
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'artifact'))
import engine_comparisons as comparisons
import engine_replication as replication

Graph, Entry = comparisons.Graph, comparisons.Entry
rank, oracle, hop_audit = comparisons.rank, comparisons.oracle, comparisons.hop_audit
METHODS = replication.METHODS + ['EagerIndexLRU', 'AdaptiveGate']


class EagerIndexLRU(comparisons.EvidenceCache):
    """Fine certificates with immediate invalidation via a reverse index.

Entry charge includes 8 bytes per reverse posting. Each nonempty index bucket
costs 24 bytes; a nonempty index has a 64-byte root. Hash allocation overhead is
not represented by this logical packed charge (see cache_snapshot).
"""
    def __init__(self, graph, capacity, epoch_fast_path=True, profile=False,
                 hybrid_tau=64):
        super().__init__(graph, 'LocalLRU', capacity, epoch_fast_path,
                         profile, hybrid_tau)
        self.method = 'EagerIndexLRU'
        self.reverse = {}
        self.index_bytes = 0
        self._notified_epoch = graph.graph_epoch
        self.stats.update(index_affected_entries=0, index_update_buckets=0,
                          index_postings_added=0, index_postings_removed=0,
                          missed_notification_flushes=0)

    @staticmethod
    def charge(key, endpoints, deps):
        # Posting references an existing resident key, rather than copying it.
        return comparisons.EvidenceCache.charge(key, endpoints, deps) + 8*len(deps)

    def drop(self, key, reason):
        entry = self.entries.pop(key)
        self.used -= entry.size
        for bucket in entry.deps:
            posting = self.reverse[bucket]
            posting.remove(key)
            self.stats['index_postings_removed'] += 1
            if not posting:
                del self.reverse[bucket]
                self.index_bytes -= 24
                self.used -= 24
        if not self.reverse and self.index_bytes:
            self.index_bytes -= 64
            self.used -= 64
        if reason is not None:
            self.stats[reason] += 1

    def _clear(self):
        # All owned posting lists disappear together, including their charges.
        self.stats['invalidated'] += len(self.entries)
        self.stats['index_postings_removed'] += sum(len(e.deps) for e in self.entries.values())
        self.entries.clear()
        self.reverse.clear()
        self.used = self.index_bytes = 0
        self.stats['flushes'] += 1

    def on_graph_update(self, bucket):
        if self.g.graph_epoch != self._notified_epoch+1:
            self._clear()
            self.stats['missed_notification_flushes'] += 1
        self.stats['index_update_buckets'] += 1
        affected = tuple(self.reverse.get(bucket, ()))
        self.stats['index_affected_entries'] += len(affected)
        for key in affected:
            self.drop(key, 'invalidated')
        self._notified_epoch = self.g.graph_epoch

    def synchronize(self):
        if self._notified_epoch != self.g.graph_epoch:
            self._clear()
            self.stats['missed_notification_flushes'] += 1
            self._notified_epoch = self.g.graph_epoch
        super().synchronize()

    def put(self, key, endpoints, deps, cost):
        if any(bucket < 0 for bucket in deps):
            raise ValueError('Reverse baseline requires fine bucket certificates')
        size = self.charge(key, endpoints, deps)
        if size > self.capacity:
            self.stats['rejected'] += 1
            return
        if key in self.entries:
            self.drop(key, None)
        while True:
            new_buckets = sum(bucket not in self.reverse for bucket in deps)
            extra = 24*new_buckets + (64 if deps and not self.reverse else 0)
            if self.used + size + extra <= self.capacity:
                break
            if not self.entries:
                self.stats['rejected'] += 1
                return
            self.drop(next(iter(self.entries)), 'evicted')
        self.clock += 1
        self.entries[key] = Entry(frozenset(endpoints), dict(deps), cost, size,
                                 touched=self.clock, validated_epoch=self.g.graph_epoch)
        self.used += size + extra
        self.index_bytes += extra
        for bucket in deps:
            self.reverse.setdefault(bucket, set()).add(key)
            self.stats['index_postings_added'] += 1
        self.peak = max(self.peak, self.used)
        assert self.used <= self.capacity


def apply_edge(cache, relation, src, dst, insert):
    """Common mutation API; returned bool excludes ineffective/no-op edits."""
    changed = cache.g.edge(relation, src, dst, insert)
    if changed and hasattr(cache, 'on_graph_update'):
        cache.on_graph_update(relation*cache.g.n+src)
    return changed


_DIAG_KEYS = ('lookup_attempts', 'resident_probes', 'epoch_fast_path_hits',
              'index_valid_hits', 'validation_calls', 'validation_failures',
              'validation_tokens', 'validation_ns', 'final_hits', 'prefix_hits',
              'query_misses', 'queries_after_graph_update',
              'graph_updates_since_previous_query')


class _DiagnosticMixin:
    """Faithful get() with diagnostic-only loop clocks and request counters."""
    def _init_diagnostics(self):
        self.diagnostic_totals = dict.fromkeys(_DIAG_KEYS, 0)
        self.last_diagnostics = dict.fromkeys(_DIAG_KEYS, 0)
        self._diag_previous_query_graph = self.g.graph_epoch
        self._diag_attempt = self._diag_hit_count = 0

    def _count(self, key, value=1):
        self.last_diagnostics[key] += value
        self.diagnostic_totals[key] += value

    def query(self, anchor, path, q, k=10):
        self.last_diagnostics = dict.fromkeys(_DIAG_KEYS, 0)
        self._diag_attempt = self._diag_hit_count = 0
        changed = self.g.graph_epoch != self._diag_previous_query_graph
        if changed:
            self._count('queries_after_graph_update')
            self._count('graph_updates_since_previous_query',
                        self.g.graph_epoch-self._diag_previous_query_graph)
        result = super().query(anchor, path, q, k)
        if not self._diag_hit_count and self.method not in ('Eager', 'EagerSorted', 'EagerMemo'):
            self._count('query_misses')
        self._diag_previous_query_graph = self.g.graph_epoch
        return result

    def get(self, key):
        self._diag_attempt += 1
        self._count('lookup_attempts')
        e = self.entries.get(key)
        if e is None:
            self.stats['misses'] += 1
            return None
        self._count('resident_probes')
        if self.method.startswith('Local'):
            if self.epoch_fast_path and e.validated_epoch == self.g.graph_epoch:
                self._count('epoch_fast_path_hits')
            else:
                self._count('validation_calls')
                begin, checked, valid = perf_counter_ns(), 0, True
                for bucket, version in e.deps.items():
                    checked += 1
                    self.stats['validated_buckets'] += 1
                    current = (self.g.versions[bucket] if bucket >= 0 else
                               self.g.relation_epochs[-1-bucket])
                    if current != version:
                        valid = False
                        break
                elapsed = perf_counter_ns()-begin
                self._count('validation_tokens', checked)
                self._count('validation_ns', elapsed)
                if not valid:
                    self._count('validation_failures')
                    self.drop(key, 'invalidated')
                    self.stats['misses'] += 1
                    return None
                e.validated_epoch = self.g.graph_epoch
        elif self.method == 'EagerIndexLRU':
            self._count('index_valid_hits')
        self.clock += 1
        e.touched = self.clock
        e.frequency += 1
        self.entries.move_to_end(key)
        self.stats['hits'] += 1
        self._diag_hit_count += 1
        self._count('final_hits' if self._diag_attempt == 1 else 'prefix_hits')
        return e


_DIAGNOSTIC_CLASSES = {}


def _diagnostic_instance(base):
    cls = type(base)
    if cls not in _DIAGNOSTIC_CLASSES:
        _DIAGNOSTIC_CLASSES[cls] = type('Diagnostic'+cls.__name__,
                                      (_DiagnosticMixin, cls), {})
    obj = object.__new__(_DIAGNOSTIC_CLASSES[cls])
    obj.__dict__ = base.__dict__
    obj._init_diagnostics()
    return obj


def EvidenceCache(graph, method, capacity, epoch_fast_path=True, profile=False,
                  hybrid_tau=64, diagnostics=False, gate_seed=0,
                  gate_warmup=32, gate_period=256, gate_block=8,
                  gate_margin=0.05, gate_alpha=0.5):
    if method == 'EagerIndexLRU':
        cache = EagerIndexLRU(graph, capacity, epoch_fast_path, profile, hybrid_tau)
    elif method == 'AdaptiveGate':
        if diagnostics or profile:
            raise ValueError('Gate exposes paid probe diagnostics, not phase profiling')
        return AdaptiveGate(graph, capacity, gate_seed, gate_warmup, gate_period,
                            gate_block, gate_margin, gate_alpha)
    else:
        cache = replication.EvidenceCache(graph, method, capacity, epoch_fast_path,
                                 profile, hybrid_tau)
    return _diagnostic_instance(cache) if diagnostics else cache


def overlap_audit(cache, changed_buckets):
    """Untimed BEFORE-update inventory: distinct changed buckets vs read domain.

For hybrid certificates relation tokens cover all graph buckets of that relation;
this reports conservative coverage, not an unavailable exact historical read set.
Stale yet resident entries are included. No versions or cache state are changed.
"""
    entries = cache.entries
    changed = set(changed_buckets)
    fine, coarse = set(), set()
    affected = 0
    for e in entries.values():
        fine.update(b for b in e.deps if b >= 0)
        coarse.update(-1-b for b in e.deps if b < 0)
        affected += any(b in e.deps or -1-(b//cache.g.n) in e.deps for b in changed)
    covered = sum(b in fine or b//cache.g.n in coarse for b in changed)
    union = len(coarse)*cache.g.n + sum(b//cache.g.n not in coarse for b in fine)
    return dict(changed_bucket_count=len(changed), overlap_bucket_count=covered,
                overlap_fraction=covered/len(changed) if changed else 0.0,
                resident_domain_buckets=union, affected_resident_entries=affected,
                resident_entries=len(entries), coarse_relations=len(coarse),
                coverage_kind='conservative_token_domain' if coarse else 'fine_read_domain')


def _deep_size(root, seen):
    if id(root) in seen:
        return 0
    seen.add(id(root))
    total = sys.getsizeof(root)
    if isinstance(root, dict):
        total += sum(_deep_size(k, seen)+_deep_size(v, seen) for k, v in root.items())
    elif isinstance(root, (tuple, list, set, frozenset)):
        total += sum(_deep_size(x, seen) for x in root)
    elif isinstance(root, Entry):
        total += _deep_size(vars(root), seen)
    return total


def cache_snapshot(cache, footprint=False):
    """Untimed resident inventory. Footprint is reachable CPython allocation size.

Objects reachable through entries and reverse index are counted once by identity;
graph/vectors, transient workspace and policy/diagnostic counters are excluded.
This is neither exclusive incremental allocation nor process RSS. Primitive
objects may also be referenced by the graph or interpreter intern table.
"""
    target = cache.cache if isinstance(cache, AdaptiveGate) else cache
    count = len(target.entries)
    charge = sum(e.size for e in target.entries.values())
    deps = sum(len(e.deps) for e in target.entries.values())
    endpoints = sum(len(e.endpoints) for e in target.entries.values())
    index = getattr(target, 'index_bytes', 0)
    result = dict(resident_entries=count, used_bytes=cache.used,
                  entry_charge_bytes=charge, index_bytes=index,
                  gate_bytes=getattr(cache, 'GATE_BYTES', 0),
                  certificate_tokens=deps, certificate_bytes=16*deps,
                  endpoint_ids=endpoints, mean_entry_charge=charge/count if count else 0.0,
                  mean_resident_charge=target.used/count if count else 0.0,
                  mean_certificate_tokens=deps/count if count else 0.0,
                  mean_endpoint_count=endpoints/count if count else 0.0,
                  lookup_hits=target.stats['hits'], lookup_misses=target.stats['misses'])
    if footprint:
        seen = set()
        physical = _deep_size(target.entries, seen)
        if hasattr(target, 'reverse'):
            physical += _deep_size(target.reverse, seen)
        result.update(resident_python_bytes=physical,
                      python_to_logical_ratio=physical/target.used if target.used else None,
                      footprint_kind='reachable_deduplicated_getsizeof_excluding_graph')
    return result


def independent_rank(vectors, endpoints, q, k):
    """Scalar Python reference, used in tests (not bitwise-equivalent reduction).

Integer/exactly representable test vectors avoid reduction-rounding ambiguity.
For arbitrary float workloads compare distances with a tolerance before claiming
independence at near ties; the primary oracle retains the original numeric kernel.
"""
    if k < 0:
        raise ValueError('k must be nonnegative')
    values = [(sum((float(x)-float(y))**2 for x, y in zip(vectors[v], q)), int(v))
              for v in endpoints]
    return tuple(v for _, v in sorted(values)[:k])


class AdaptiveGate:
    """Paid causal probe selector, not an oracle or optimal admission policy."""
    GATE_BYTES = 256

    def __init__(self, graph, capacity, seed=0, warmup=32, period=256,
                 block=8, margin=0.05, alpha=0.5):
        if capacity < self.GATE_BYTES:
            raise ValueError('AdaptiveGate requires at least 256 bytes for gate state')
        if warmup < 1 or not 1 <= block < period or not 0 <= margin < 1 or not 0 < alpha <= 1:
            raise ValueError('Invalid causal gate parameters')
        self.g, self.method, self.capacity = graph, 'AdaptiveGate', capacity
        self.cache = replication.EvidenceCache(graph, 'LocalLRUMemo', capacity-self.GATE_BYTES)
        self.memo = replication.EvidenceCache(graph, 'EagerMemo', 0)
        self.parameters = dict(seed=seed, warmup=warmup, period=period, block=block,
                               margin=margin, alpha=alpha, metadata_bytes=self.GATE_BYTES)
        self.request = self._probe_count = self._cache_sum = self._memo_sum = 0
        self.cache_estimate = self.memo_estimate = None
        self.selected = 'EagerMemo'
        self.last_gate = {}
        self.last_profile_ns = {}
        self.last_dependency_count = self.last_endpoint_count = 0
        self.last_memo_peak_bytes = self.last_memo_entries = 0
        self.stats = dict.fromkeys(self.cache.stats, 0)
        self.stats.update(gate_probes=0, gate_blocks=0, gate_switches=0,
                          gate_cache_queries=0, gate_memo_queries=0,
                          gate_probe_elapsed_ns=0)

    @property
    def entries(self):
        return self.cache.entries

    @property
    def used(self):
        return self.GATE_BYTES+self.cache.used

    @property
    def peak(self):
        return self.GATE_BYTES+self.cache.peak

    def synchronize(self):
        # Dormant cache is retained and its graph versions remain authoritative.
        self.cache.synchronize()
        self.memo.synchronize()

    def query(self, anchor, path, q, k=10):
        i, p = self.request, self.parameters
        probe = (i < p['warmup'] or
                 (i >= p['warmup']+p['period'] and
                  (i-p['warmup']) % p['period'] < p['block']))
        self.request += 1
        previous = self.selected
        before = {name: self.cache.stats[name]+self.memo.stats[name] for name in self.cache.stats}
        elapsed = {}
        if probe:
            order = (('LocalLRUMemo', self.cache), ('EagerMemo', self.memo))
            if (p['seed']+i) % 2:
                order = tuple(reversed(order))
            outputs = {}
            for name, executor in order:
                begin = perf_counter_ns()
                outputs[name] = executor.query(anchor, path, q, k)
                elapsed[name] = perf_counter_ns()-begin
            # Equality is checked by the separate oracle harness, not re-scanned here.
            result = outputs[previous]
            self._cache_sum += elapsed['LocalLRUMemo']
            self._memo_sum += elapsed['EagerMemo']
            self._probe_count += 1
            self.stats['gate_probes'] += 1
            self.stats['gate_probe_elapsed_ns'] += sum(elapsed.values())
            target_count = p['warmup'] if i < p['warmup'] else p['block']
            if self._probe_count == target_count:
                c, m = self._cache_sum/self._probe_count, self._memo_sum/self._probe_count
                self.cache_estimate = c if self.cache_estimate is None else p['alpha']*c+(1-p['alpha'])*self.cache_estimate
                self.memo_estimate = m if self.memo_estimate is None else p['alpha']*m+(1-p['alpha'])*self.memo_estimate
                self.selected = ('LocalLRUMemo' if self.cache_estimate <= (1-p['margin'])*self.memo_estimate else 'EagerMemo')
                self.stats['gate_blocks'] += 1
                self.stats['gate_switches'] += self.selected != previous
                self._probe_count = self._cache_sum = self._memo_sum = 0
        else:
            executor = self.cache if previous == 'LocalLRUMemo' else self.memo
            result = executor.query(anchor, path, q, k)
        active = self.cache if previous == 'LocalLRUMemo' else self.memo
        self.stats['gate_cache_queries' if previous == 'LocalLRUMemo' else 'gate_memo_queries'] += 1
        for name in self.cache.stats:
            self.stats[name] += self.cache.stats[name]+self.memo.stats[name]-before[name]
        for name in ('last_dependency_count', 'last_endpoint_count', 'last_memo_peak_bytes', 'last_memo_entries'):
            setattr(self, name, getattr(active, name))
        if probe:
            # Both candidate maps are discarded before their query returns.
            # Their peaks are sequential; result buffers are a separate scope.
            self.last_memo_peak_bytes = max(self.cache.last_memo_peak_bytes,
                                            self.memo.last_memo_peak_bytes)
            self.last_memo_entries = max(self.cache.last_memo_entries,
                                         self.memo.last_memo_entries)
        self.last_gate = dict(request=i+1, probe=probe, served_by=previous,
                              next_selected=self.selected, cache_estimate_ns=self.cache_estimate,
                              memo_estimate_ns=self.memo_estimate, **elapsed)
        return result

"""Read-only reanalysis of the frozen main gate evidence for adaptive routing.

No engine execution, parameter fitting, or counterfactual policy replay occurs.
Only the selected held-out raw files are rehashed against the full main strict
manifest; derived main CSVs are rehashed against the final analysis audit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MAIN = HERE.parent.parent / "experiments"
SEEDS = (11, 29, 47, 61, 73)
METHODS = ("AdaptiveGate", "EagerMemo", "LocalLRUMemo")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def flag(value):
    if value not in ("True", "False"):
        raise ValueError("Invalid recorded boolean: " + str(value))
    return value == "True"


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-6), (a, b)


def decomposition(gate, memo, cache, extra):
    best = min(memo, cache)
    if min(best, gate) <= 0:
        raise ValueError("Nonpositive service denominator")
    excess = (gate - best) / best
    paid = extra / best
    return dict(excess=excess, paid=paid, remainder=excess-paid,
                paid_over_gate=extra/gate)


def confusion(predicted, observed):
    c = Counter(("T" if p == o else "F") + ("P" if p else "N")
                for p, o in zip(predicted, observed))
    n = len(predicted)
    tp, tn, fp, fn = (c[k] for k in ("TP", "TN", "FP", "FN"))
    return dict(n=n, TP=tp, TN=tn, FP=fp, FN=fn,
                accuracy=(tp+tn)/n, positive_prevalence=(tp+fn)/n,
                always_cache_accuracy=(tp+fn)/n, always_memo_accuracy=(tn+fp)/n,
                balanced_accuracy=((tp/(tp+fn)+tn/(tn+fp))/2
                                   if tp+fn and tn+fp else None))


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0]*len(values)
    start = 0
    while start < len(order):
        end = start+1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for i in order[start:end]:
            result[i] = (start+end-1)/2
        start = end
    return result


def pearson(x, y):
    if len(x) < 3 or max(x) == min(x) or max(y) == min(y):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def correlation(x, y):
    if not all(v > 0 for v in x+y):
        raise ValueError("Log-ratio correlation requires positive ratios")
    return dict(n=len(x), pearson_log_ratio=pearson([math.log(v) for v in x],
                                                  [math.log(v) for v in y]),
                spearman_ratio=pearson(ranks(x), ranks(y)))


def longest_true(values):
    longest = run = 0
    for v in values:
        run = run+1 if v else 0
        longest = max(longest, run)
    return longest


def routes(ledger, ids):
    selected = [ledger[i]["served_by"] == "LocalLRUMemo" for i in ids]
    probes = [flag(ledger[i]["probe"]) for i in ids]
    n = len(ids)
    dormant = [not s and not p for s, p in zip(selected, probes)]
    return dict(requests=n, probes=sum(probes),
                cache_served_fraction=sum(selected)/n,
                memo_served_fraction=1-sum(selected)/n,
                probe_fraction=sum(probes)/n,
                cache_candidate_executed_fraction=sum(s or p for s,p in zip(selected,probes))/n,
                cache_served_nonprobe_fraction=sum(s and not p for s,p in zip(selected,probes))/(n-sum(probes)),
                dormant_requests=sum(dormant),
                longest_dormant_nonprobe_streak=longest_true(dormant))


def process_ratio(rows, numerator, denominator):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["process_repeat"]].append(r)
    values = [sum(r[numerator] for r in rr)/sum(r[denominator] for r in rr)
              for _,rr in sorted(grouped.items())]
    assert len(values) == 3
    return dict(mean=st.mean(values), minimum=min(values), maximum=max(values),
                process_values=values)


def run(main, out):
    inputs = {}
    def bind(path, expected=None):
        digest = sha(path)
        if expected is not None:
            assert digest == expected, "Input hash mismatch: " + str(path)
        inputs[str(path.relative_to(main)).replace("\\", "/")] = digest
        return path

    aa = json.loads(bind(main/"analysis_final/analysis_audit.json").read_text())
    strict = json.loads(bind(main/"analysis_final/strict_integrity.json").read_text())
    raw = json.loads(bind(main/"audit_main.json").read_text())
    assert aa["partial"] is False and aa["strict_integrity_status"] == "PASS"
    assert aa["method_executions"] == 5999680 and not aa["failures"]
    assert strict["status"] == "PASS" and not strict["failures"]
    assert raw["status"] == "PASS"
    assert sha(main/"audit_main.json") == aa["raw_audit_sha256"]
    def derived(name):
        return read_csv(bind(main/"analysis_final"/name, aa["output_sha256"][name]))
    def frozen(rel):
        return bind(main/rel, strict["files"][rel])
    for name in ("engine_main.py","engine_protocol.md",
                 "case_config.py","case_config_frozen.json"):
        frozen(name)

    overhead = [r for r in derived("gate_overhead.csv")
                if r["suite"] == "grid" and r["split"] == "heldout"]
    pilots = [r for r in derived("pilot_predictions.csv")
              if r["suite"] == "grid" and r["split"] == "heldout"]
    blocks = [r for r in derived("gate_blocks.csv")
              if r["suite"] == "grid" and r["split"] == "heldout"]
    derived("predictor_classification.csv")
    # Registry supplies the already published Cxxx label; it does not select cells.
    registry = read_csv(bind(main/"generated_assets/csv/configuration_ids.csv"))
    labels = {r["case_id"]:r["id"] for r in registry if r["suite"] == "grid"}
    config_keys = sorted({(r["suite"],r["case_id"]) for r in derived("configuration_summary.csv")})
    regenerated_labels = {cid:f"C{i+1:03d}" for i,(suite,cid) in enumerate(config_keys) if suite=="grid"}
    assert labels == regenerated_labels
    idx = {(r["case_id"],int(r["process_repeat"]),int(r["seed"]),r["segment"]):r
           for r in overhead}
    pidx = {(r["case_id"],int(r["process_repeat"]),int(r["seed"])):r for r in pilots}
    bidx = {(r["case_id"],int(r["process_repeat"]),int(r["seed"]),int(r["block"])):r
            for r in blocks}
    assert len(pidx) == 240 and len(bidx) == 480
    cells = sorted({r["case_id"] for r in pilots},
                   key=lambda c:(int(next(r["degree"] for r in pilots if r["case_id"]==c)),
                                 int(next(r["length"] for r in pilots if r["case_id"]==c)),
                                 int(next(r["budget"] for r in pilots if r["case_id"]==c))))
    assert len(cells) == 16
    runs, forecasts = [], []
    for rep in range(3):
        frozen(f"results/grid_p{rep}/environment.json")
    for cid in cells:
        for rep in range(3):
            for seed in SEEDS:
                key = cid, rep, seed
                traces, metas = {}, {}
                for method in METHODS:
                    stem = f"results/grid_p{rep}/{cid}_s{seed}_{method}"
                    meta = json.loads(frozen(stem+".json").read_text())
                    assert (meta["case_id"],meta["process_repeat"],meta["seed"],meta["method"]) == (*key,method)
                    assert meta["diagnostic"] is False and meta["suite"] == "grid"
                    metas[method] = meta
                    traces[method] = read_csv(frozen(stem+"_traces.csv"))
                assert len({m["workload_sha256"] for m in metas.values()}) == 1
                ledger = read_csv(frozen(f"results/grid_p{rep}/{cid}_s{seed}_AdaptiveGate_gate.csv"))
                assert len(ledger) == 512 and all(len(t)==512 for t in traces.values())
                assert all(int(row["qid"])==i for rows in [ledger,*traces.values()]
                           for i,row in enumerate(rows))
                c = metas["AdaptiveGate"]["case"]
                assert ([4,8,16,32].index(c["degree"])+[3,4,5,6].index(c["length"])
                        +[65536,1048576].index(c["budget"])) % 2 == 1
                base = dict(case_id=cid,label=labels[cid],degree=c["degree"],length=c["length"],
                            budget_bytes=c["budget"],process_repeat=rep,seed=seed)
                for segment,ids in [("all",list(range(512))),("post_pilot",list(range(32,512)))]:
                    totals = {m:sum(float(traces[m][i]["service_us"]) for i in ids) for m in METHODS}
                    extra = sum(float(ledger[i]["EagerMemo" if ledger[i]["served_by"]=="LocalLRUMemo" else "LocalLRUMemo"])/1000
                                for i in ids if flag(ledger[i]["probe"]))
                    d = decomposition(totals["AdaptiveGate"],totals["EagerMemo"],totals["LocalLRUMemo"],extra)
                    reference = idx[key+(segment,)]
                    for m,field in [("AdaptiveGate","gate_total_us"),("EagerMemo","memo_total_us"),("LocalLRUMemo","cache_total_us")]:
                        close(totals[m],reference[field])
                    close(extra,reference["paid_extra_evaluation_us"])
                    rt = routes(ledger,ids)
                    close(rt["cache_served_fraction"],reference["served_cache_fraction"])
                    assert rt["probes"] == (40 if segment=="all" else 8)
                    dormant_ids=[i for i in ids if ledger[i]["served_by"]=="EagerMemo" and not flag(ledger[i]["probe"])]
                    runs.append(dict(**base,segment=segment,**rt,
                                     gate_total_us=totals["AdaptiveGate"],memo_total_us=totals["EagerMemo"],cache_total_us=totals["LocalLRUMemo"],
                                     extra_candidate_us=extra,**d,
                                     dormant_effective_edits=sum(int(traces["AdaptiveGate"][i]["effective_edits"]) for i in dormant_ids),
                                     dormant_mean_resident_entries=(st.mean(float(traces["AdaptiveGate"][i]["resident_entries"]) for i in dormant_ids) if dormant_ids else None)))
                for target,block,decision,ids in [("pilot_steady",0,31,list(range(128,512))),
                                                   ("next_nonprobe_block",0,31,list(range(32,288))),
                                                   ("next_nonprobe_block",1,295,list(range(296,512)))]:
                    row=ledger[decision]
                    ce,me=float(row["cache_estimate_ns"]),float(row["memo_estimate_ns"])
                    predicted,selected=ce<me,ce<=.95*me
                    assert selected == (row["next_selected"]=="LocalLRUMemo")
                    sums={m:sum(float(traces[m][i]["service_us"]) for i in ids) for m in METHODS}
                    observed=sums["LocalLRUMemo"]<sums["EagerMemo"]
                    reference=pidx[key] if target=="pilot_steady" else bidx[key+(block,)]
                    close(me/ce,reference["pilot_memo_over_cache"] if target=="pilot_steady"
                          else float(reference["memo_estimate_ns"])/float(reference["cache_estimate_ns"]))
                    close(sums["EagerMemo"]/sums["LocalLRUMemo"],reference["static_steady_memo_over_cache"] if target=="pilot_steady" else reference["static_future_memo_over_cache"])
                    assert predicted == flag(reference["predicted_cache_faster"])
                    assert observed == flag(reference["observed_static_cache_faster"])
                    forecasts.append(dict(**base,target=target,block=block,decision_qid=decision,
                                          target_start=ids[0],target_stop_exclusive=ids[-1]+1,
                                          forecast_memo_over_cache=me/ce,
                                          realized_memo_over_cache=sums["EagerMemo"]/sums["LocalLRUMemo"],
                                          realized_memo_over_gate=sums["EagerMemo"]/sums["AdaptiveGate"],
                                          realized_cache_over_gate=sums["LocalLRUMemo"]/sums["AdaptiveGate"],
                                          sign_cache=predicted,selected_cache=selected,observed_cache_faster=observed))

    summaries=[]
    for cid in cells:
        rr=[r for r in runs if r["case_id"]==cid]
        base={k:rr[0][k] for k in ["case_id","label","degree","length","budget_bytes"]}
        result=dict(**base,runs=15)
        for segment in ("all","post_pilot"):
            ss=[r for r in rr if r["segment"]==segment]
            for field in ["excess","paid","remainder","paid_over_gate","cache_served_fraction","memo_served_fraction",
                          "probe_fraction","cache_candidate_executed_fraction","cache_served_nonprobe_fraction",
                          "dormant_requests","longest_dormant_nonprobe_streak","dormant_effective_edits"]:
                result[segment+"_"+field]=st.mean(r[field] for r in ss)
            for numerator,denominator,name in [("memo_total_us","gate_total_us","memo_over_gate"),
                                               ("cache_total_us","gate_total_us","cache_over_gate"),
                                               ("memo_total_us","cache_total_us","memo_over_cache")]:
                for k,v in process_ratio(ss,numerator,denominator).items():
                    result[segment+"_"+name+"_"+k]=json.dumps(v) if isinstance(v,list) else v
        summaries.append(result)
    classifications=[]
    for target in ("pilot_steady","next_nonprobe_block"):
        rows=[r for r in forecasts if r["target"]==target]
        for classifier in ("sign_cache","selected_cache"):
            classifications.append(dict(target=target,classifier=classifier,
                                        **confusion([r[classifier] for r in rows],[r["observed_cache_faster"] for r in rows]),
                                        dependence="16 fixed cells; five reused seeds x three process repeats; paired forecasts, not independent trials"))
    correlations=[]
    for target in ("pilot_steady","next_nonprobe_block"):
        rows=[r for r in forecasts if r["target"]==target]
        for outcome in ("realized_memo_over_cache","realized_memo_over_gate","realized_cache_over_gate"):
            for cid in cells:
                rr=[r for r in rows if r["case_id"]==cid]
                correlations.append(dict(target=target,outcome=outcome,scope="within_cell",case_id=cid,label=labels[cid],
                                         **correlation([r["forecast_memo_over_cache"] for r in rr],[r[outcome] for r in rr])))
            # Between-cell correlations use geometric means (mean log ratios),
            # so each cell receives one point regardless of repeated observations.
            x,y=[],[]
            for cid in cells:
                rr=[r for r in rows if r["case_id"]==cid]
                x.append(math.exp(st.mean(math.log(r["forecast_memo_over_cache"]) for r in rr)))
                y.append(math.exp(st.mean(math.log(r[outcome]) for r in rr)))
            loo=[correlation(x[:i]+x[i+1:],y[:i]+y[i+1:])["pearson_log_ratio"] for i in range(len(x))]
            correlations.append(dict(target=target,outcome=outcome,scope="between_cell_geometric_means",case_id="ALL",label="ALL",
                                     **correlation(x,y),leave_one_cell_out_pearson_min=min(loo),leave_one_cell_out_pearson_max=max(loo)))
    q3={}
    for segment in ("post_pilot","all"):
        for threshold,name in [(1.25,"cache_cost_at_least_20pct_lower"),(1.20,"cache_speedup_over_1p20")]:
            rr=[r for r in summaries if r[segment+"_memo_over_cache_mean"]>threshold]
            q3[segment+"_"+name]=dict(cells=len(rr),labels=[r["label"] for r in rr],
                                     definition=f"Cell mean of three matched-seed-mean static Memo/cache ratios > {threshold}",
                                     post_pilot_cache_served_fraction=st.mean(r["post_pilot_cache_served_fraction"] for r in rr) if rr else None,
                                     per_cell_post_pilot_cache_served_fraction={r["label"]:r["post_pilot_cache_served_fraction"] for r in rr})
    facts=dict(status="PASS",heldout_cells=16,paired_runs=240,post_pilot_requests_per_run=480,
               post_pilot_probe_requests_per_run=8,
               macro_decomposition={segment:{k:st.mean(r[segment+"_"+k] for r in summaries)
                                             for k in ("excess","paid","remainder","paid_over_gate")}
                                    for segment in ("all","post_pilot")},
               post_pilot_cache_served_fraction=st.mean(r["post_pilot_cache_served_fraction"] for r in summaries),
               post_pilot_cache_candidate_executed_fraction=st.mean(r["post_pilot_cache_candidate_executed_fraction"] for r in summaries),
               classification=classifications,q3=q3,
               between_cell_correlations=[r for r in correlations if r["scope"]!="within_cell"],
               sign_to_margin_flips={target:{"positive_to_memo":sum(r["sign_cache"] and not r["selected_cache"] for r in forecasts if r["target"]==target),
                                            "true_cache_advantage":sum(r["sign_cache"] and not r["selected_cache"] and r["observed_cache_faster"] for r in forecasts if r["target"]==target),
                                            "true_memo_advantage":sum(r["sign_cache"] and not r["selected_cache"] and not r["observed_cache_faster"] for r in forecasts if r["target"]==target)}
                                     for target in ("pilot_steady","next_nonprobe_block")})
    close(facts["macro_decomposition"]["all"]["excess"],0.27086966782952643)
    assert all(math.isclose(r["excess"],r["paid"]+r["remainder"],abs_tol=1e-12) for r in runs)
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in [("cell_summary",summaries),("paired_runs",runs),("forecasts",forecasts),
                      ("classification",classifications),("correlations",correlations)]:
        write_csv(out/(name+".csv"),rows)
    (out/"key_facts.json").write_text(json.dumps(facts,indent=2),encoding="utf-8")
    make_tex(out,summaries,correlations,facts)
    (out/"input_manifest.json").write_text(json.dumps(dict(main_files_sha256=inputs,
        frozen_raw_binding="Every selected raw file is rehashed against analysis_final/strict_integrity.json",
        derived_binding="Selected main CSVs rehashed against analysis_final/analysis_audit.json output_sha256",
        script_sha256=sha(Path(__file__)),
        definition_tests_sha256=sha(HERE/"test_gate_analysis.py")),indent=2),encoding="utf-8")
    audit=dict(status="PASS",runtime=dict(python=sys.version,numpy=np.__version__,platform=platform.platform()),
               method_runs_reused=720,raw_method_request_rows_checked=368640,gate_ledger_rows_checked=122880,
               new_benchmarks=0,heldout_cells=16,paired_runs=240,input_files=len(inputs),
               outputs_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!="analysis_audit.json"},
               limitations=["Fixed grid, shared seeds and repeated processes; correlations are descriptive, with no iid p-values or confidence intervals",
                            "Arithmetic remainder is not an identified causal state-decay cost",
                            "Threshold comparison evaluates recorded forecasts only, not counterfactual route histories or service",
                            "No symmetric-margin, budget-release, or refresh policy experiment was conducted"])
    (out/"analysis_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(json.dumps(dict(status="PASS",cells=len(summaries),decomposition=facts["macro_decomposition"],q3=q3),indent=2))


def make_tex(out, cells, correlations, facts):
    lines=[r"% Generated from frozen main evidence by analyses/gate/analyze_gate_analysis.py.",
           r"\subsection{Gate routes, forecasts, and complete-service cost}\label{analysis-gate-detail}",
           r"All 16 held-out grid cells contain the same five seeds in three fresh processes. Post-pilot means requests 33--512 (zero-based qids 32--511), including 8 periodic probe requests; both candidates execute on these probes. A served route denotes the candidate whose result is returned, while candidate execution also includes an unselected candidate on a probe. These are request fractions, not lookup hit rates.",
           r"\begin{table*}[t]\centering\footnotesize",
           r"\caption{Post-pilot returned routes and complete-service ratios. $M$ is EagerMemo, $C$ is LocalLRU+memo, and $G$ is AdaptiveGate. Cache/Memo columns are complementary served-route percentages; Exec. counts all cache executions. Probe fraction is 1.67\% in every cell. Ratios are means of three process-level ratios of matched five-seed means; parentheses give their observed minima/maxima, not confidence intervals. A comparator/$G$ ratio above one favors the gate.}\label{analysis-gate-routes}",
           r"\begin{tabular}{lrrr rrr ll}\hline",
           r"Cell & Deg. & Hops & KiB & Cache\% & Memo\% & Exec.\% & $M/G$ & $C/G$\\\hline"]
    for r in cells:
        ratios=[f"{r['post_pilot_'+k+'_mean']:.3f} ({r['post_pilot_'+k+'_minimum']:.3f},{r['post_pilot_'+k+'_maximum']:.3f})" for k in ("memo_over_gate","cache_over_gate")]
        lines.append(f"{r['label']} & {r['degree']} & {r['length']} & {r['budget_bytes']//1024} & {100*r['post_pilot_cache_served_fraction']:.1f} & {100*r['post_pilot_memo_served_fraction']:.1f} & {100*r['post_pilot_cache_candidate_executed_fraction']:.1f} & {ratios[0]} & {ratios[1]} "+r"\\")
    lines += [r"\hline\end{tabular}\end{table*}",
              r"For a matched seed/process run, let $G,M,C$ be all-512-request complete-service totals, $B=\min(M,C)$, and $X$ the sum of the unselected candidate's timed evaluation on probe requests. The execution order alternates, so this extra evaluation need not occur second. Define $E=(G-B)/B$, $P=X/B$, and $R=E-P$. We first average each quantity across the 15 paired runs within a cell and then equally across cells. Thus $\overline E=\overline P+\overline R$ preserves the original 27.09\% statistic; $X/G$ uses a different denominator. The best-static denominator is selected per paired run, not from pooled cell totals.",
              r"\begin{table*}[t]\centering\footnotesize",
              r"\caption{All-request per-cell gate excess and its arithmetic decomposition, including the initial paid pilot. $E,P,R$ use the same matched-run best-static denominator; $X/G$ is the separately reported fraction of gate service. All entries are percentages. The remainder is not a measured state-decay component.}\label{analysis-gate-overhead}",
              r"\begin{tabular}{lrrr rrrr}\hline Cell & Deg. & Hops & KiB & $E$ & $P$ & $R$ & $X/G$\\\hline"]
    for r in cells:
        values=" & ".join(f"{100*r['all_'+k]:.2f}" for k in ("excess","paid","remainder","paid_over_gate"))
        lines.append(f"{r['label']} & {r['degree']} & {r['length']} & {r['budget_bytes']//1024} & {values} "+r"\\")
    z=facts["macro_decomposition"]["all"]
    lines += [r"\hline Mean & & & & "+" & ".join(f"{100*z[k]:.2f}" for k in ("excess","paid","remainder","paid_over_gate"))+r"\\\hline\end{tabular}\end{table*}",
              r"The residual combines unallocated routing/orchestration cost, different cache histories, the gate's 256-byte reserve, comparison to continuously active static controls, and timing variation. Dormancy (Memo-only non-probe requests) is observable in the ledger, but the cost of a counterfactual continuously refreshed cache is not. Neither the sign nor the size of a causal state-decay contribution can be recovered by subtracting measured probe work.",
              r"\begin{table*}[t]\centering\footnotesize",
              r"\caption{Within-cell descriptive forecast/realization correlations. $r$ is Pearson correlation of natural-log ratios, and $\rho$ is Spearman correlation. Pilot forecasts target the static $M/C$ ratio over requests 129--512 (15 paired observations/cell). Block forecasts target the following non-probe window, requests 33--288 or 297--512 (30 observations/cell). The last column targets actual gate advantage $M/G$ in those same block windows. Shared seeds, repeated processes, and successive blocks are dependent; no iid significance test is applied.}\label{analysis-gate-correlations}",
              r"\begin{tabular}{lrrrrr}\hline Cell & Pilot $r$ & Pilot $\rho$ & Block $r$ & Block $\rho$ & Block $r$ for $M/G$\\\hline"]
    for r in cells:
        def get(target,outcome):
            return next(c for c in correlations if c["case_id"]==r["case_id"] and c["target"]==target and c["outcome"]==outcome)
        p=get("pilot_steady","realized_memo_over_cache");b=get("next_nonprobe_block","realized_memo_over_cache");g=get("next_nonprobe_block","realized_memo_over_gate")
        lines.append(r["label"]+" & "+" & ".join("--" if v is None else f"{v:.3f}" for v in (p["pearson_log_ratio"],p["spearman_ratio"],b["pearson_log_ratio"],b["spearman_ratio"],g["pearson_log_ratio"]))+r"\\")
    lines += [r"\hline\end{tabular}\end{table*}",
              r"The forecast ratio measures paid candidate evaluation time on the gate's own history; the realized targets include complete service on independently executed static histories or the actual gate history, as named. Forecast and outcome are dimensionless ratios, not absolute microseconds. The CSV also reports between-cell correlations of geometric-mean ratios (one point per cell) and leave-one-cell-out sensitivity ranges; these describe this fixed grid, not a population sample."]
    (out/"gate_supplement.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")
    q=facts["q3"]["post_pilot_cache_cost_at_least_20pct_lower"]
    route=100*facts["post_pilot_cache_served_fraction"]
    main=(r"The gate returns the cached candidate on "+f"{route:.1f}"+r"\% of post-pilot requests across the 16 held-out cells; both candidates additionally execute on the 1.67\% periodic probes (Table~\ref{analysis-gate-routes}). "
          +f"In the {q['cells']} cells with a post-pilot static Memo/cache ratio above 1.25, its cache-served fraction is {100*q['post_pilot_cache_served_fraction']:.1f}"+r"\%. "
          +r"The all-request 27.09\% mean cell excess over the matched best static route decomposes into 10.44 percentage points of extra candidate evaluation and a 16.65-point arithmetic remainder on that same denominator (Table~\ref{analysis-gate-overhead}); the previously reported 8.26\% divides extra evaluation by gate service instead. The remainder does not identify state decay. The reported pilot confusion matrix uses a sign threshold of one; the actual 0.95 rule yields TP/TN/FP/FN of 78/88/7/67, versus 101/85/10/44 for that sign classifier on the same targets. The fixed 5\% margin is a prespecified heuristic favoring Memo, not a fitted optimum or symmetric hysteresis. A symmetric margin, budget release during dormancy, and paid refresh/admission on Memo-served requests remain untested controls. A refresh policy would have to charge its work and retain complete certificates and exact validation; its service cannot be inferred from these traces."
          +"\n")
    (out/"gate_main_paragraph.tex").write_text(main,encoding="utf-8")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--main",type=Path,default=MAIN)
    parser.add_argument("--out",type=Path,default=HERE/"results")
    args=parser.parse_args()
    if args.main.resolve() in args.out.resolve().parents or args.main.resolve()==args.out.resolve():
        raise SystemExit("Refusing to write into immutable main")
    run(args.main.resolve(),args.out.resolve())

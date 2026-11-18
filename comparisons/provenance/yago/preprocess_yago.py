#!/usr/bin/env python3
"""Convert a YAGO 4.5 Turtle zip into SAGO CSV format.

The converter uses real YAGO RDF triples. It maps all entities to a coarse
`Entity` label and stores RDF classes as SAGO node types, which creates the
schema-rich/coarse-label setting where SAGO is intended to help.
"""

import argparse
import csv
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
ENTITY_PREFIXES = (
    "http://yago-knowledge.org/resource/",
    "http://www.wikidata.org/entity/",
)


def clean(value):
    return (value or "").strip().replace(",", " ").replace("|", " ").replace(";", " ")


def local_name(uri):
    uri = uri.strip("<>")
    if "#" in uri:
        uri = uri.rsplit("#", 1)[1]
    else:
        uri = uri.rstrip("/").rsplit("/", 1)[-1]
    return clean(uri) or "Thing"


def is_entity(uri):
    return uri.startswith(ENTITY_PREFIXES)


def expand(term, prefixes):
    term = term.strip()
    if term.startswith("<") and term.endswith(">"):
        return term[1:-1]
    if ":" in term and not term.startswith(("http://", "https://")):
        pfx, rest = term.split(":", 1)
        if pfx in prefixes:
            return prefixes[pfx] + rest
    return term


def tokenize_triple(line):
    line = line.strip()
    if not line or line.startswith("#") or not line.endswith("."):
        return None
    line = line[:-1].strip()
    tokens = []
    i = 0
    while i < len(line):
        while i < len(line) and line[i].isspace():
            i += 1
        if i >= len(line):
            break
        if line[i] == "<":
            j = line.find(">", i + 1)
            if j < 0:
                return None
            tokens.append(line[i : j + 1])
            i = j + 1
        elif line[i] == '"':
            return None
        else:
            j = i
            while j < len(line) and not line[j].isspace():
                j += 1
            tokens.append(line[i:j])
            i = j
        if len(tokens) == 3:
            break
    return tokens if len(tokens) == 3 else None


def iter_triples(zip_path):
    prefixes = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "schema": "http://schema.org/",
        "yago": "http://yago-knowledge.org/resource/",
    }
    prefix_re = re.compile(r"@prefix\s+([^:\s]+):\s+<([^>]+)>")
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith((".ttl", ".nt"))]
        for name in names:
            with zf.open(name) as fh:
                for raw in fh:
                    line = raw.decode("utf-8", errors="ignore").strip()
                    m = prefix_re.match(line)
                    if m:
                        prefixes[m.group(1)] = m.group(2)
                        continue
                    parts = tokenize_triple(line)
                    if not parts:
                        continue
                    s, p, o = (expand(x, prefixes) for x in parts)
                    if o.startswith('"'):
                        continue
                    yield s, p, o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw/yago/yago-4.5.0.2-tiny.zip")
    ap.add_argument("--out", default="datasets/yago")
    ap.add_argument("--max-entities", type=int, default=200000)
    ap.add_argument("--max-edges", type=int, default=1000000)
    ap.add_argument("--max-types-per-node", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    entity_id = {}
    node_types = defaultdict(list)
    raw_edges = []

    def get_entity(uri):
        if not is_entity(uri):
            return None
        if uri not in entity_id and len(entity_id) < args.max_entities:
            entity_id[uri] = len(entity_id)
        return entity_id.get(uri)

    # First pass: collect entity ids and RDF types.
    for s, p, o in iter_triples(args.raw):
        if p == RDF_TYPE:
            sid = get_entity(s)
            if sid is not None and len(node_types[sid]) < args.max_types_per_node:
                t = local_name(o)
                if t not in node_types[sid]:
                    node_types[sid].append(t)
        elif is_entity(o):
            get_entity(s)
            get_entity(o)
        if len(entity_id) >= args.max_entities:
            break

    # Second pass: collect relation edges among retained entities.
    for s, p, o in iter_triples(args.raw):
        if p == RDF_TYPE or s not in entity_id or o not in entity_id:
            continue
        etype = local_name(p)
        if etype in {"label", "comment", "sameAs"}:
            continue
        raw_edges.append((entity_id[s], entity_id[o], etype, etype))
        if len(raw_edges) >= args.max_edges:
            break

    nodes = []
    for uri, nid in sorted(entity_id.items(), key=lambda kv: kv[1]):
        types = node_types[nid] or ["Thing"]
        nodes.append((nid, "Entity", "|".join(types), "name=" + local_name(uri)))

    rel_src = defaultdict(Counter)
    rel_dst = defaultdict(Counter)
    for s, d, _, etype in raw_edges:
        for t in node_types[s] or ["Thing"]:
            rel_src[etype][t] += 1
        for t in node_types[d] or ["Thing"]:
            rel_dst[etype][t] += 1

    with (out / "nodes.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "label", "types", "props"])
        w.writerows(nodes)

    with (out / "edges.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["src", "dst", "label", "type"])
        w.writerows(raw_edges)

    with (out / "schema_edges.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["type", "label", "src_types", "dst_types"])
        for etype in sorted(rel_src):
            src_types = sorted(rel_src[etype]) or ["Thing"]
            dst_types = sorted(rel_dst[etype]) or ["Thing"]
            w.writerow([etype, etype, "|".join(src_types), "|".join(dst_types)])

    print(f"nodes={len(nodes)} edges={len(raw_edges)} edge_types={len(rel_src)}")


if __name__ == "__main__":
    main()

"""Builds a directed call graph (NetworkX) from a RepoIndex.

Calls are resolved by NAME (static, heuristic). Python is dynamic, so an edge means
"this function contains a call whose name matches that function", not a proof that
the call really reaches it at runtime. Edges with several candidates are flagged
as ambiguous.
"""
from __future__ import annotations

from itertools import islice

import networkx as nx

from scanner import FunctionInfo, RepoIndex


def build_call_graph(index: RepoIndex) -> nx.DiGraph:
    g = nx.DiGraph()
    by_name: dict[str, list[FunctionInfo]] = {}
    for f in index.functions.values():
        g.add_node(f.id, file=f.file, name=f.name, line=f.start_line, kind=f.kind)
        if f.kind != "module":
            by_name.setdefault(f.name, []).append(f)

    for caller in index.functions.values():
        for call in caller.calls:
            candidates = by_name.get(call.name, [])
            if not candidates:
                continue  # builtin / library call; not part of this repository
            chosen = candidates
            is_self_call = call.text.startswith(("self.", "cls."))
            if is_self_call and caller.class_name:
                same_class = [c for c in candidates
                              if c.file == caller.file and c.class_name == caller.class_name]
                chosen = same_class or candidates
            elif "." not in call.text:
                same_file = [c for c in candidates if c.file == caller.file]
                chosen = same_file or candidates
            ambiguous = len(chosen) > 1
            for callee in chosen:
                if g.has_edge(caller.id, callee.id):
                    g[caller.id][callee.id]["lines"].append(call.line)
                    g[caller.id][callee.id]["ambiguous"] |= ambiguous
                else:
                    g.add_edge(caller.id, callee.id, lines=[call.line], ambiguous=ambiguous)
    return g


def callers_of(g: nx.DiGraph, node_id: str) -> list[str]:
    return sorted(g.predecessors(node_id)) if node_id in g else []


def callees_of(g: nx.DiGraph, node_id: str) -> list[str]:
    return sorted(g.successors(node_id)) if node_id in g else []


def find_paths(g: nx.DiGraph, src: str, dst: str, max_paths: int = 3, cutoff: int = 10) -> list[list[str]]:
    """Shortest-first simple paths. Simple paths never revisit a node, so cycles and
    recursion cannot loop forever."""
    if src not in g or dst not in g or src == dst:
        return []
    paths = nx.all_simple_paths(g, src, dst, cutoff=cutoff)
    return sorted(islice(paths, 200), key=len)[:max_paths]


def find_cycles(g: nx.DiGraph, limit: int = 10) -> list[list[str]]:
    """Recursion (self-loops) and mutual recursion, capped so huge graphs stay fast."""
    return list(islice(nx.simple_cycles(g), limit))


def reachable_from(g: nx.DiGraph, node_id: str) -> set[str]:
    return set(nx.descendants(g, node_id)) if node_id in g else set()

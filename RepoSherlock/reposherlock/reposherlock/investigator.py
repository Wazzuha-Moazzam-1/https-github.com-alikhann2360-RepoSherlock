"""Investigator: decides WHICH code is relevant to a question and computes the facts
that can be verified from static analysis alone (no AI involved). The AI layer only
sees what this module selects.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import networkx as nx

from graph_builder import callers_of, callees_of, find_cycles, find_paths
from scanner import FunctionInfo, RepoIndex

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in", "on", "for", "and", "or",
    "how", "what", "which", "who", "where", "why", "when", "does", "do", "did", "this", "that",
    "it", "its", "work", "works", "project", "repository", "repo", "code", "function", "functions",
    "explain", "summarize", "summary", "describe", "show", "me", "with", "from", "by", "as", "at",
    "might", "could", "can", "responsible", "call", "calls", "called", "flow", "execution",
}
ARCH_WORDS = {"architecture", "overview", "summarize", "summary", "structure", "organized", "organised"}
BUG_WORDS = {"error", "exception", "traceback", "bug", "fail", "fails", "failing", "crash", "crashes",
             "keyerror", "typeerror", "valueerror", "attributeerror", "indexerror", "nameerror"}


@dataclass
class Location:
    file: str
    start: int
    end: int

    def __str__(self) -> str:
        return format_location(self.file, self.start, self.end)


def format_location(file: str, start: int, end: int | None = None) -> str:
    if end is None or end == start:
        return f"{file}:{start}"
    return f"{file}:{start}-{end}"


@dataclass
class Relevant:
    func: FunctionInfo
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class Fact:
    text: str
    locations: list[Location] = field(default_factory=list)


@dataclass
class Snippet:
    sid: str                 # "S1", "S2", ... the only IDs the model may cite
    func: FunctionInfo
    start_line: int
    lines: list[str]
    truncated: bool

    def numbered(self) -> str:
        return "\n".join(f"{self.start_line + i:>4} | {line}" for i, line in enumerate(self.lines))


@dataclass
class Investigation:
    question: str
    intent: str
    relevant: list[Relevant] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
    snippets: list[Snippet] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------- helpers

def split_identifier(text: str) -> list[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return [t for t in re.split(r"[^A-Za-z0-9]+", text.lower()) if t]


def question_tokens(question: str) -> list[str]:
    seen: list[str] = []
    for tok in split_identifier(question):
        if tok not in STOPWORDS and tok not in BUG_WORDS and len(tok) > 1 and tok not in seen:
            seen.append(tok)
    return seen


def _related(a: str, b: str) -> bool:
    if a == b:
        return True
    return min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a))


def resolve_name(index: RepoIndex, name: str) -> list[FunctionInfo]:
    name = name.strip().rstrip("()")
    return [f for f in index.real_functions() if f.name == name or f.qualname == name]


def function_at(index: RepoIndex, file_hint: str, line: int) -> FunctionInfo | None:
    """Innermost function containing `line` in a file whose path ends with `file_hint`."""
    hint = file_hint.replace("\\", "/")
    best = None
    for f in index.real_functions():
        if (f.file == hint or f.file.endswith("/" + hint.split("/")[-1])) and f.start_line <= line <= f.end_line:
            if best is None or (f.end_line - f.start_line) < (best.end_line - best.start_line):
                best = f
    return best


def _loc(f: FunctionInfo) -> Location:
    return Location(f.file, f.start_line, f.end_line)


def _label(f: FunctionInfo) -> str:
    return f"{f.qualname}() [{format_location(f.file, f.start_line, f.end_line)}]"


def _fn(index: RepoIndex, node_id: str) -> FunctionInfo | None:
    return index.functions.get(node_id)


# ----------------------------------------------------------------------------- intent

def detect_intent(question: str) -> tuple[str, dict]:
    q = question.strip()
    ql = q.lower()
    m = re.search(r"from\s+`?([A-Za-z_][\w.]*)(?:\(\))?`?\s+to\s+`?([A-Za-z_][\w.]*)(?:\(\))?`?", q)
    if m:
        return "flow", {"src": m.group(1), "dst": m.group(2)}
    m = re.search(r"(?:call(?:s|ed|ers?)?(?:\s+of)?|invoke[sd]?|use[sd]?)\s+`?([A-Za-z_][\w.]*)\(\)`?", q) \
        or re.search(r"\bcall(?:s|ed|ers?)?\s+(?:of\s+)?`([A-Za-z_][\w.]*)`", q)
    if m and ("call" in ql or "invoke" in ql):
        return "callers", {"name": m.group(1)}
    words = set(split_identifier(q))
    if words & BUG_WORDS or "traceback" in ql:
        return "bug", {}
    if words & ARCH_WORDS:
        return "architecture", {}
    return "explain", {}


# ----------------------------------------------------------------------------- scoring

def score_functions(index: RepoIndex, question: str, limit: int = 6) -> list[Relevant]:
    tokens = question_tokens(question)
    mentioned = set(re.findall(r"`([A-Za-z_][\w.]*)`|\b([A-Za-z_]\w*)\(\)", question))
    mentioned_names = {a or b for a, b in mentioned}
    # also treat any word exactly equal to a function name as a mention
    known = {f.name for f in index.real_functions()}
    for w in re.findall(r"[A-Za-z_]\w*", question):
        if w in known and "_" in w:
            mentioned_names.add(w)

    results: list[Relevant] = []
    for f in index.real_functions():
        score, reasons = 0.0, []
        if f.name in mentioned_names or f.qualname in mentioned_names:
            score += 10
            reasons.append("named in the question")
        name_parts = split_identifier(f.name)
        hits = [t for t in tokens if any(_related(t, p) for p in name_parts)]
        if hits:
            score += 3 * len(hits)
            reasons.append("name matches: " + ", ".join(hits))
        path_parts = split_identifier(f.file)
        phits = [t for t in tokens if any(_related(t, p) for p in path_parts)]
        if phits:
            score += 1.5 * len(phits)
            reasons.append("file path matches: " + ", ".join(phits))
        doc_parts = split_identifier(f.docstring)
        dhits = [t for t in tokens if any(_related(t, p) for p in doc_parts)]
        if dhits:
            score += 1.5 * len(dhits)
            reasons.append("docstring matches: " + ", ".join(dhits))
        body = " ".join(index.lines.get(f.file, [])[f.start_line - 1:f.end_line]).lower()
        bhits = [t for t in tokens if len(t) >= 4 and t in body]
        if bhits:
            score += min(2.0, 0.5 * len(bhits))
            reasons.append("body mentions: " + ", ".join(bhits[:3]))
        if score > 0:
            results.append(Relevant(f, score, reasons))
    results.sort(key=lambda r: (-r.score, r.func.file, r.func.start_line))
    return results[:limit]


def _expand_with_neighbors(index: RepoIndex, g: nx.DiGraph, picked: list[Relevant], max_total: int) -> list[Relevant]:
    have = {r.func.id for r in picked}
    out = list(picked)
    for r in list(picked[:3]):
        for nid in callees_of(g, r.func.id) + callers_of(g, r.func.id):
            f = _fn(index, nid)
            if f is None or f.kind == "module" or nid in have or len(out) >= max_total:
                continue
            relation = "called by" if nid in g.predecessors(r.func.id) else "calls"
            # relation text is from the neighbour's point of view
            out.append(Relevant(f, r.score * 0.3, [f"{'calls' if relation == 'called by' else 'is called by'} {r.func.name}()"]))
            have.add(nid)
    return out


# ----------------------------------------------------------------------------- intents

def _edge_lines(g: nx.DiGraph, a: str, b: str) -> list[int]:
    return sorted(set(g[a][b]["lines"])) if g.has_edge(a, b) else []


def _callers_facts(index, g, name) -> tuple[list[Relevant], list[Fact], list[str]]:
    targets = resolve_name(index, name)
    if not targets:
        return [], [], [f"No function named '{name}' was found in the repository."]
    rel, facts, lims = [], [], []
    for t in targets:
        rel.append(Relevant(t, 100, ["target of the question"]))
        callers = callers_of(g, t.id)
        if not callers:
            facts.append(Fact(f"No call to {t.qualname}() was found by static analysis.", [_loc(t)]))
            lims.append("'No callers' only covers direct, name-based calls; dynamic dispatch, callbacks, "
                        "decorators and framework hooks are invisible to this analysis.")
            continue
        parts, locs = [], [_loc(t)]
        for cid in callers:
            c = _fn(index, cid)
            if c is None:
                continue
            lines = _edge_lines(g, cid, t.id)
            where = "module level" if c.kind == "module" else f"{c.qualname}()"
            parts.append(f"{where} in {c.file} at line(s) {', '.join(map(str, lines))}"
                         + (" (name match is ambiguous)" if g[cid][t.id]["ambiguous"] else ""))
            locs.append(Location(c.file, lines[0], lines[-1]))
            if c.kind != "module":
                rel.append(Relevant(c, 50, [f"calls {t.name}()"]))
        facts.append(Fact(f"{t.qualname}() is called from: " + "; ".join(parts) + ".", locs))
    return rel, facts, lims


def _flow_facts(index, g, src_name, dst_name) -> tuple[list[Relevant], list[Fact], list[str]]:
    srcs, dsts = resolve_name(index, src_name), resolve_name(index, dst_name)
    lims = []
    if not srcs:
        lims.append(f"No function named '{src_name}' was found.")
    if not dsts:
        lims.append(f"No function named '{dst_name}' was found.")
    if not srcs or not dsts:
        return [], [], lims
    rel, facts, seen = [], [], set()
    found_any = False
    for s in srcs:
        for d in dsts:
            paths = find_paths(g, s.id, d.id)
            for p in paths:
                found_any = True
                names = [(_fn(index, n).qualname + "()") for n in p]
                hops, locs = [], []
                for a, b in zip(p, p[1:]):
                    hops.append(f"{_fn(index, a).qualname}() calls {_fn(index, b).qualname}() at "
                                f"{_fn(index, a).file}:{', '.join(map(str, _edge_lines(g, a, b)))}")
                    locs.append(Location(_fn(index, a).file, _edge_lines(g, a, b)[0], _edge_lines(g, a, b)[-1]))
                facts.append(Fact("Static call path: " + " -> ".join(names) + ". " + "; ".join(hops) + ".", locs))
                for n in p:
                    f = _fn(index, n)
                    if f and f.kind != "module" and n not in seen:
                        seen.add(n)
                        rel.append(Relevant(f, 80, ["on a call path"]))
    if not found_any:
        facts.append(Fact(f"No static call path from {src_name}() to {dst_name}() was found in the call graph."))
        lims.append("A missing path does not prove the functions are unrelated; dynamic calls are not visible.")
        rel = [Relevant(f, 60, ["endpoint of the requested flow"]) for f in srcs + dsts]
    return rel, facts, lims


def _architecture_facts(index, g) -> tuple[list[Relevant], list[Fact], list[str]]:
    facts = []
    funcs = index.real_functions()
    facts.append(Fact(f"The repository has {len(index.files)} Python file(s), {len(funcs)} function(s)/method(s) "
                      f"and {len(index.classes)} class(es)."))
    per_file: dict[str, int] = {}
    for f in funcs:
        per_file[f.file] = per_file.get(f.file, 0) + 1
    if per_file:
        top = sorted(per_file.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
        facts.append(Fact("Files by number of functions: " + ", ".join(f"{k} ({v})" for k, v in top) + "."))
    entries = [f for f in funcs if f.name == "main"]
    for m in index.functions.values():
        if m.kind == "module" and m.calls:
            facts.append(Fact(f"Module-level code in {m.file} calls: "
                              + ", ".join(sorted({c.name + '()' for c in m.calls})[:8]) + ".",
                              [Location(m.file, m.calls[0].line, m.calls[-1].line)]))
    hubs = sorted((f for f in funcs), key=lambda f: -(g.in_degree(f.id) + g.out_degree(f.id)) if f.id in g else 0)[:6]
    hubs = [h for h in hubs if h.id in g and g.degree(h.id) > 0]
    if hubs:
        facts.append(Fact("Most connected functions: " + ", ".join(
            f"{h.qualname}() (called by {g.in_degree(h.id)}, calls {g.out_degree(h.id)})" for h in hubs) + ".",
            [_loc(h) for h in hubs]))
    cycles = find_cycles(g, 5)
    if cycles:
        facts.append(Fact("Recursive/mutually-recursive call cycles: " + "; ".join(
            " -> ".join(_fn(index, n).qualname + "()" for n in c + [c[0]]) for c in cycles) + ".",
            [_loc(_fn(index, c[0])) for c in cycles if _fn(index, c[0]) and _fn(index, c[0]).kind != "module"]))
    rel = [Relevant(f, 20, ["entry point (main)"]) for f in entries[:2]]
    rel += [Relevant(h, 10, ["highly connected"]) for h in hubs if h not in entries]
    return rel[:8], facts, []


_TRACE_RE = re.compile(r'File "([^"]+)", line (\d+)(?:, in (\S+))?')


def _bug_facts(index, question) -> tuple[list[Relevant], list[Fact], list[str]]:
    rel, facts, lims = [], [], []
    for path, line, _name in _TRACE_RE.findall(question):
        f = function_at(index, path, int(line))
        if f:
            facts.append(Fact(f"The traceback frame {path.replace(chr(92), '/')}:{line} lies inside {f.qualname}().",
                              [_loc(f)]))
            rel.append(Relevant(f, 90, ["named in the traceback"]))
    literals = [a or b for a, b in re.findall(r"'([^'\n]{2,60})'|\"([^\"\n]{2,60})\"", question)]
    for lit in literals[:3]:
        hits = [(file, i + 1) for file, lines in index.lines.items() for i, ln in enumerate(lines) if lit in ln]
        if hits:
            facts.append(Fact(f"The text '{lit}' appears in the source at: "
                              + ", ".join(format_location(f, n) for f, n in hits[:8]) + ".",
                              [Location(f, n, n) for f, n in hits[:8]]))
            for file, n in hits[:4]:
                fn = function_at(index, file, n)
                if fn:
                    rel.append(Relevant(fn, 70, [f"contains the text '{lit}'"]))
        else:
            facts.append(Fact(f"The text '{lit}' does not appear anywhere in the scanned Python source."))
        # Leads (not facts): functions whose bodies contain words related to the literal,
        # e.g. 'db_url' -> 'url' also appears in a config key built elsewhere.
        for part in [p for p in split_identifier(lit) if len(p) >= 3 and p not in STOPWORDS][:3]:
            added = 0
            for f in index.real_functions():
                body = "\n".join(index.lines.get(f.file, [])[f.start_line - 1:f.end_line]).lower()
                if part in body and not any(x.func.id == f.id for x in rel) and added < 3:
                    rel.append(Relevant(f, 40, [f"body contains '{part}', related to '{lit}' (lead, unverified)"]))
                    added += 1
    lims.append("Bug analysis is a set of leads, not a diagnosis: nothing was executed, so no failure was reproduced.")
    return rel, facts, lims


# ----------------------------------------------------------------------------- main entry

def build_snippets(index: RepoIndex, relevant: list[Relevant], max_lines: int = 40,
                   budget_chars: int = 7000) -> list[Snippet]:
    snippets, used = [], 0
    for r in relevant:
        start, lines, truncated = index.get_source(r.func, max_lines)
        cost = sum(len(x) for x in lines)
        if snippets and used + cost > budget_chars:
            break
        snippets.append(Snippet(f"S{len(snippets) + 1}", r.func, start, lines, truncated))
        used += cost
    return snippets


def investigate(index: RepoIndex, g: nx.DiGraph, question: str, max_functions: int = 8) -> Investigation:
    intent, args = detect_intent(question)
    inv = Investigation(question=question, intent=intent)
    rel: list[Relevant] = []
    if intent == "callers":
        rel, inv.facts, inv.limitations = _callers_facts(index, g, args["name"])
    elif intent == "flow":
        rel, inv.facts, inv.limitations = _flow_facts(index, g, args["src"], args["dst"])
    elif intent == "architecture":
        rel, inv.facts, inv.limitations = _architecture_facts(index, g)
    elif intent == "bug":
        rel, inv.facts, inv.limitations = _bug_facts(index, question)

    scored = score_functions(index, question)
    if intent in ("callers", "flow") and rel:
        scored = []   # the graph already gave exact answers; fuzzy keyword matches would only add noise
    have = {r.func.id for r in rel}
    rel += [r for r in scored if r.func.id not in have]
    if not rel:
        inv.limitations.append("No functions matched the words in the question; try naming a function, "
                               "file or concept that appears in the code.")
        hubs = sorted(index.real_functions(), key=lambda f: -(g.degree(f.id) if f.id in g else 0))[:3]
        rel = [Relevant(h, 1, ["fallback: highly connected function"]) for h in hubs]
    if intent in ("explain", "bug"):
        rel = _expand_with_neighbors(index, g, rel, max_functions)
    # de-duplicate, keep order
    seen, uniq = set(), []
    for r in rel:
        if r.func.id not in seen:
            seen.add(r.func.id)
            uniq.append(r)
    inv.relevant = uniq[:max_functions]
    inv.snippets = build_snippets(index, inv.relevant)
    if any(s.truncated for s in inv.snippets):
        inv.limitations.append("Some long functions were truncated before being sent to the model.")
    if len(inv.snippets) < len(inv.relevant):
        inv.limitations.append("Some relevant functions were left out to keep the model prompt small.")
    if index.warnings:
        inv.limitations.append(f"{len(index.warnings)} scan warning(s), e.g. {index.warnings[0]}")
    inv.limitations.append("The call graph is built from name matches (static analysis). Edges are leads, "
                           "not proof of runtime behaviour.")
    # keep order stable and remove duplicates
    inv.limitations = list(dict.fromkeys(inv.limitations))
    return inv

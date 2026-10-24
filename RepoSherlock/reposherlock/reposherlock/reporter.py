"""Turns an investigation into an evidence-backed report (Markdown). The same text is
rendered in the terminal with Rich and can be saved to a .md file."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import networkx as nx

from ai_analyst import AIResult
from investigator import Investigation, Snippet, format_location
from notifier import DeliveryResult
from scanner import RepoIndex

STATUS_TEXT = {
    "confirmed": "CONFIRMED - received back through DNotifier",
    "sent_unconfirmed": "SENT, NOT CONFIRMED - handed to DNotifier, no echo seen",
    "failed": "FAILED",
    "skipped": "NOT SENT - DNotifier not used",
}


@dataclass
class Report:
    inv: Investigation
    index: RepoIndex
    graph: nx.DiGraph
    ai: AIResult | None = None
    ai_error: str | None = None
    deliveries: list[DeliveryResult] = field(default_factory=list)
    dnotifier_state: str = ""


def format_evidence(snippet: Snippet, include_code: bool = True) -> str:
    """One evidence entry: id, function, exact location, and (optionally) numbered source."""
    f = snippet.func
    end = snippet.start_line + len(snippet.lines) - 1
    head = f"**[{snippet.sid}]** `{f.qualname}()` - `{format_location(f.file, snippet.start_line, end)}`"
    if snippet.truncated:
        head += " (truncated)"
    if not include_code:
        return head
    return f"{head}\n\n```python\n{snippet.numbered()}\n```"


def call_relationships(index: RepoIndex, g: nx.DiGraph, inv: Investigation, limit: int = 14) -> list[str]:
    ids = {r.func.id for r in inv.relevant}
    out = []
    for a, b, data in g.edges(data=True):
        if a in ids or b in ids:
            fa, fb = index.functions[a], index.functions[b]
            who = "module level" if fa.kind == "module" else f"{fa.qualname}()"
            lines = ", ".join(str(n) for n in sorted(set(data["lines"])))
            note = " (name match is ambiguous)" if data.get("ambiguous") else ""
            out.append(f"{who} -> {fb.qualname}() at `{fa.file}:{lines}`{note}")
    return out[:limit]


def short_answer(r: Report) -> str:
    if r.ai and r.ai.answer:
        return r.ai.answer
    names = ", ".join(f"{x.func.qualname}()" for x in r.inv.relevant[:5]) or "none"
    prefix = f"AI explanation unavailable ({r.ai_error}). " if r.ai_error else "AI explanation was skipped. "
    return prefix + f"Static analysis selected these functions as most relevant: {names}."


def to_markdown(r: Report, include_code: bool = True) -> str:
    inv = r.inv
    out: list[str] = [f"# RepoSherlock report", "",
                      f"*Generated {datetime.now():%Y-%m-%d %H:%M} - repository: `{r.index.root}`*", "",
                      "## Question", "", inv.question, "",
                      "## Short answer", "", short_answer(r), ""]

    out += ["## Relevant functions", ""]
    if inv.relevant:
        for x in inv.relevant:
            f = x.func
            out.append(f"- `{f.qualname}()` - `{format_location(f.file, f.start_line, f.end_line)}` "
                       f"- why: {'; '.join(x.reasons)}")
    else:
        out.append("- none found")
    out.append("")

    rels = call_relationships(r.index, r.graph, inv)
    out += ["## Call relationships (static, name-based)", ""]
    out += [f"- {x}" for x in rels] or ["- no calls between the relevant functions were found"]
    out.append("")

    out += ["## Verified facts (computed from the source, no AI involved)", ""]
    for fact in inv.facts:
        locs = ", ".join(f"`{loc}`" for loc in fact.locations[:6])
        out.append(f"- {fact.text}" + (f" Evidence: {locs}" if locs else ""))
    if not inv.facts:
        out.append("- none computed for this kind of question")
    out.append("")

    out += ["## Hypotheses (unverified - from the local AI model)", ""]
    if r.ai and r.ai.hypotheses:
        out += [f"- {h}" for h in r.ai.hypotheses]
    else:
        out.append("- none")
    out.append("")

    out += ["## Evidence", ""]
    for s in inv.snippets:
        out += [format_evidence(s, include_code), ""]
    if not inv.snippets:
        out += ["No source snippets were selected.", ""]

    out += ["## Limitations", ""]
    lims = list(inv.limitations)
    if r.ai:
        lims.append(f"Explanation generated locally by `{r.ai.model}` through Ollama; small models can be wrong.")
        lims += [f"WARNING: {w}" for w in r.ai.warnings]
    out += [f"- {x}" for x in lims]
    out.append("")

    out += ["## DNotifier event delivery", ""]
    if r.deliveries:
        for d in r.deliveries:
            lat = f" ({d.latency_ms} ms)" if d.latency_ms is not None else ""
            extra = f" - {d.detail}" if d.detail and d.status != "confirmed" else ""
            out.append(f"- `{d.event}`: {STATUS_TEXT.get(d.status, d.status)}{lat}{extra}")
        if any(not d.delivered for d in r.deliveries):
            out.append("- Events that were not confirmed were saved to the LOCAL-ONLY file "
                       "`reposherlock_events.local.jsonl`. That file is a fallback, not DNotifier delivery.")
    else:
        out.append(f"- No events were sent. {r.dnotifier_state}".rstrip())
    out.append("")
    return "\n".join(out)

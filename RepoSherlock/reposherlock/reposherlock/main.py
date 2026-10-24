"""RepoSherlock - terminal AI agent that investigates unfamiliar Python codebases.

  python main.py analyze <repo>
  python main.py ask <repo> "How does authentication work?"
  python main.py functions <repo>
  python main.py doctor            # check Ollama + DNotifier setup
  python main.py monitor           # live view of events (run in a 2nd terminal)
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):          # avoid cp1252 crashes in Windows PowerShell
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.table import Table

load_dotenv(Path(__file__).resolve().parent / ".env")   # works no matter where you run it from

import ai_analyst
import notifier as dn
from graph_builder import build_call_graph, find_cycles
from investigator import format_location, investigate
from reporter import Report, to_markdown
from scanner import scan_repository

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="RepoSherlock: ask questions about a Python codebase, with evidence.")
console = Console()

RepoArg = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, resolve_path=True,
                         help="Path to the Python repository (a folder).")
NoDN = typer.Option(False, "--no-dnotifier", help="LOCAL-ONLY run: do not use DNotifier.")


def ok(msg: str) -> None:
    console.print(f"[green][OK][/green] {escape(msg)}")


def warn(msg: str) -> None:
    console.print(f"[yellow][WARN][/yellow] {escape(msg)}")


def fail(msg: str) -> None:
    console.print(f"[red][ERROR][/red] {escape(msg)}")


def show_delivery(res: dn.DeliveryResult) -> None:
    if res.delivered:
        ok(f"DNotifier event '{res.event}' confirmed ({res.latency_ms} ms)")
    elif res.status == "skipped":
        warn(f"DNotifier event '{res.event}' NOT sent: {res.detail} (saved to local-only file)")
    else:
        warn(f"DNotifier event '{res.event}' {res.status}: {res.detail} (saved to local-only file)")


def start_notifier(disabled: bool) -> dn.Notifier:
    n = dn.Notifier(enabled=not disabled)
    if disabled:
        warn("DNotifier disabled (--no-dnotifier). This is a LOCAL-ONLY run.")
        return n
    with console.status("Connecting to DNotifier..."):
        ready, detail = n.start()
    if ready:
        ok("Connected to DNotifier (agent + delivery-check clients)")
        for note in n.notes:
            warn(note)
    else:
        warn(f"DNotifier unavailable: {detail}. Continuing LOCAL-ONLY; nothing will be claimed as delivered.")
    return n


def emit(n: dn.Notifier, event: str, summary: str, data: dict | None = None) -> dn.DeliveryResult:
    res = n.emit(event, summary, data)
    if n.state == "ready":           # disabled / unavailable were already announced once at startup
        show_delivery(res)
    return res


def scan(repo: Path):
    with console.status("Scanning repository (read-only, nothing is executed)..."):
        index = scan_repository(repo)
        graph = build_call_graph(index)
    return index, graph


# ------------------------------------------------------------------------------ analyze
@app.command()
def analyze(repo: Path = RepoArg, no_dnotifier: bool = NoDN) -> None:
    """Scan a repository and build its call graph."""
    console.print(f"[bold]RepoSherlock is analyzing[/bold] {escape(str(repo))} ...")
    n = start_notifier(no_dnotifier)
    try:
        emit(n, "investigation_started", f"Analyzing {repo.name}", {"command": "analyze", "repo": repo.name})
        index, graph = scan(repo)
        if not index.files:
            fail("No Python files found in that folder.")
            emit(n, "investigation_failed", "No Python files found", {"stage": "scan"})
            raise typer.Exit(1)
        funcs = index.real_functions()
        ok(f"Python files discovered ({len(index.files)})")
        ok(f"Functions and classes extracted ({len(funcs)} functions/methods, {len(index.classes)} classes)")
        ok(f"Call graph constructed ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)")
        for w in index.warnings[:5]:
            warn(w)
        emit(n, "repository_scanned", f"{len(index.files)} files, {len(funcs)} functions",
             {"files": len(index.files), "functions": len(funcs), "classes": len(index.classes),
              "edges": graph.number_of_edges(), "warnings": len(index.warnings)})

        cycles = find_cycles(graph, 5)
        for c in cycles[:2]:
            names = " -> ".join(index.functions[x].qualname + "()" for x in c + [c[0]])
            first = index.functions[c[0]]
            console.print(f"[cyan]Cycle found:[/cyan] {escape(names)}")
            emit(n, "finding_discovered", f"Call cycle: {names}",
                 {"kind": "call_cycle", "location": format_location(first.file, first.start_line)})

        table = Table(title="Files")
        table.add_column("File"); table.add_column("Functions", justify="right")
        counts: dict[str, int] = {}
        for f in funcs:
            counts[f.file] = counts.get(f.file, 0) + 1
        for file in index.files:
            table.add_row(escape(file), str(counts.get(file, 0)))
        console.print(table)
        emit(n, "report_completed", "Analysis finished", {"command": "analyze"})
        if n.state != "disabled":
            console.print(f"\n{n.summary_line()}")
        console.print("Next: [bold]python main.py ask <repo> \"your question\"[/bold]")
    finally:
        n.close()


# ------------------------------------------------------------------------------ ask
@app.command()
def ask(
    repo: Path = RepoArg,
    question: str = typer.Argument(..., help="Your question, in quotes."),
    save: Path = typer.Option(None, "--save", "-s", help="Also save the report as a Markdown file."),
    no_ai: bool = typer.Option(False, "--no-ai", help="Static analysis only; skip the Ollama model."),
    model: str = typer.Option(None, "--model", help="Ollama model (default qwen2.5-coder:3b)."),
    timeout: int = typer.Option(None, "--timeout", help="Seconds to wait for the model (default 180)."),
    show_code: bool = typer.Option(False, "--show-code", help="Print the evidence code in the terminal."),
    no_dnotifier: bool = NoDN,
) -> None:
    """Ask a question about a repository and get an evidence-backed report."""
    n = start_notifier(no_dnotifier)
    stage = "start"
    try:
        emit(n, "investigation_started", question[:120], {"command": "ask", "repo": repo.name,
                                                        "question": question[:200]})
        stage = "scan"
        index, graph = scan(repo)
        if not index.functions:
            raise RuntimeError("No Python functions were found in that folder.")
        ok(f"Scanned {len(index.files)} file(s), {len(index.real_functions())} function(s)")
        emit(n, "repository_scanned", f"{len(index.files)} files, {len(index.real_functions())} functions",
             {"files": len(index.files), "functions": len(index.real_functions()),
              "edges": graph.number_of_edges()})

        stage = "investigate"
        inv = investigate(index, graph, question)
        ok(f"Retrieved {len(inv.snippets)} relevant function(s) (intent: {inv.intent})")
        for r in inv.relevant[:3]:
            f = r.func
            emit(n, "finding_discovered", f"Relevant: {f.qualname}()",
                 {"function": f.qualname, "location": format_location(f.file, f.start_line, f.end_line),
                  "why": r.reasons[0] if r.reasons else ""})

        ai_result, ai_error = None, None
        if not no_ai:
            stage = "ai"
            _, mdl, to = ai_analyst.get_settings(model, timeout)
            try:
                with console.status(f"Asking local model {mdl} via Ollama (timeout {to}s)..."):
                    ai_result = ai_analyst.analyze(inv, index, model, timeout)
                ok(f"Local AI explanation generated ({mdl})")
            except ai_analyst.OllamaError as exc:
                ai_error = str(exc).splitlines()[0]
                warn(str(exc))
                warn("Continuing with static analysis only.")

        stage = "report"
        short = (ai_result.answer if ai_result else inv.facts[0].text if inv.facts else "static analysis only")
        emit(n, "report_completed", short[:160].replace("\n", " "),
             {"relevant": len(inv.relevant), "ai_used": bool(ai_result), "intent": inv.intent})

        report = Report(inv=inv, index=index, graph=graph, ai=ai_result, ai_error=ai_error,
                        deliveries=[] if n.state == "disabled" else list(n.results),
                        dnotifier_state=n.state_detail)
        console.print()
        console.print(Markdown(to_markdown(report, include_code=show_code)))
        if save:
            save.write_text(to_markdown(report, include_code=True), encoding="utf-8")
            ok(f"Report saved to {save}")
    except typer.Exit:
        raise
    except Exception as exc:
        fail(f"{exc.__class__.__name__} while {stage}: {exc}")
        emit(n, "investigation_failed", f"{exc.__class__.__name__} during {stage}",
             {"stage": stage, "error": exc.__class__.__name__})
        raise typer.Exit(1)
    finally:
        n.close()


# ------------------------------------------------------------------------------ functions
@app.command()
def functions(repo: Path = RepoArg,
              name: str = typer.Option(None, "--filter", "-f", help="Only show names containing this text.")) -> None:
    """List every function/method and class found (local only, no DNotifier)."""
    index, graph = scan(repo)
    table = Table(title=f"Functions in {repo.name}")
    for col in ("Name", "Kind", "Location", "Calls", "Called by"):
        table.add_column(col)
    for f in sorted(index.real_functions(), key=lambda x: (x.file, x.start_line)):
        if name and name.lower() not in f.qualname.lower():
            continue
        table.add_row(escape(f.qualname + "()"), f.kind, escape(format_location(f.file, f.start_line, f.end_line)),
                      str(graph.out_degree(f.id)), str(graph.in_degree(f.id)))
    console.print(table)
    if index.classes:
        ctable = Table(title="Classes")
        for col in ("Name", "Location", "Methods"):
            ctable.add_column(col)
        for c in index.classes:
            ctable.add_row(escape(c.name), escape(format_location(c.file, c.start_line, c.end_line)), str(len(c.methods)))
        console.print(ctable)
    for w in index.warnings[:5]:
        warn(w)


# ------------------------------------------------------------------------------ doctor
@app.command()
def doctor() -> None:
    """Check that Ollama and DNotifier are set up correctly."""
    console.print("[bold]RepoSherlock setup check[/bold]\n")
    host, model, _ = ai_analyst.get_settings()
    try:
        ai_analyst.ensure_model(host, model)
        ok(f"Ollama is running and model '{model}' is installed")
    except ai_analyst.OllamaError as exc:
        fail(str(exc))

    n = dn.Notifier()
    if not n._app_id or not n._secret:
        fail("DNotifier credentials missing: copy .env.example to .env and fill DNOTIFIER_APP_ID and "
             "DNOTIFIER_SECRET from https://app.dnotifier.com")
        return
    ok(f"DNotifier credentials found (app id starts with '{n._app_id[:4]}...', secret hidden)")
    with console.status("Connecting to DNotifier..."):
        ready, detail = n.start()
    if not ready:
        fail(detail)
        return
    ok("Connected and authenticated with DNotifier over WebSocket")
    if n.plan_limits:
        console.print(f"    plan limits reported by DNotifier: {escape(str(n.plan_limits))}")
    for note in n.notes:
        warn(note)
    res = n.emit("doctor_test", "RepoSherlock setup check", {"check": True})
    show_delivery(res)
    if res.delivered:
        ok("Round trip verified: an event was sent and received back through DNotifier. You are ready.")
    else:
        fail("Round trip NOT verified. Do not claim DNotifier delivery until this passes.")
    n.close()


# ------------------------------------------------------------------------------ monitor
@app.command()
def monitor(seconds: int = typer.Option(None, "--seconds", help="Stop after N seconds (default: until Ctrl+C).")) -> None:
    """Live view of RepoSherlock events arriving through DNotifier (use a second terminal)."""
    def on_event(body: dict, sender: str) -> None:
        ev = body.get("event", body.get("type", "message"))
        console.print(f"[cyan]{escape(str(ev))}[/cyan]  {escape(str(body.get('text', body))[:160])}  "
                      f"[dim]run={escape(str(body.get('run_id', '?')))}[/dim]")

    console.print("Listening for RepoSherlock events through DNotifier (Ctrl+C to stop)...")
    try:
        asyncio.run(dn.monitor_events(on_event, seconds, on_ready=lambda: ok("Monitor connected; waiting for events")))
    except KeyboardInterrupt:
        console.print("\nStopped.")
    except Exception as exc:
        fail(f"{exc.__class__.__name__}: {exc}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

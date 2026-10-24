from pathlib import Path

from graph_builder import build_call_graph
from investigator import detect_intent, format_location, investigate
from reporter import Report, format_evidence, to_markdown
from notifier import DeliveryResult
from scanner import scan_repository

SAMPLE = Path(__file__).resolve().parent.parent / "sample_project"
index = scan_repository(SAMPLE)
graph = build_call_graph(index)


def test_format_location():
    assert format_location("auth.py", 5, 9) == "auth.py:5-9"
    assert format_location("auth.py", 5, 5) == "auth.py:5"
    assert format_location("auth.py", 5) == "auth.py:5"


def test_intent_detection():
    assert detect_intent("Which functions call login()?")[0] == "callers"
    assert detect_intent("Explain the flow from main() to connect_database()") == \
        ("flow", {"src": "main", "dst": "connect_database"})
    assert detect_intent("Why does it fail with KeyError?")[0] == "bug"
    assert detect_intent("Summarize the architecture")[0] == "architecture"
    assert detect_intent("How does authentication work?")[0] == "explain"


def test_authentication_question_finds_auth_code():
    inv = investigate(index, graph, "How does authentication work in this project?")
    names = {r.func.name for r in inv.relevant}
    assert {"login", "validate_credentials"} <= names


def test_callers_fact_has_real_line_numbers():
    inv = investigate(index, graph, "Which functions call login()?")
    assert "main() in main.py at line(s) 14" in inv.facts[0].text
    assert "main.py" in {loc.file for loc in inv.facts[0].locations}


def test_bug_question_finds_literal_and_key_mismatch_lead():
    inv = investigate(index, graph, "Which functions might cause KeyError: 'db_url'?")
    assert any("database.py:17" in f.text for f in inv.facts)
    assert {"connect_database", "build_config"} <= {r.func.name for r in inv.relevant}


def test_evidence_snippet_lines_are_exact():
    inv = investigate(index, graph, "Which functions call login()?")
    snip = inv.snippets[0]
    text = format_evidence(snip)
    first = snip.start_line
    assert f"`{snip.func.file}:{first}-" in text
    for i, line in enumerate(snip.lines):
        assert f"{first + i:>4} | {line}" in text
        assert index.lines[snip.func.file][first + i - 1] == line


def test_report_contains_required_sections_and_honest_delivery_status():
    inv = investigate(index, graph, "Which functions call login()?")
    deliveries = [DeliveryResult("investigation_started", "a1", "confirmed", latency_ms=120),
                  DeliveryResult("report_completed", "b2", "sent_unconfirmed", "no echo within 8s")]
    md = to_markdown(Report(inv, index, graph, deliveries=deliveries))
    for section in ("## Question", "## Short answer", "## Relevant functions", "## Evidence",
                    "## Limitations", "## DNotifier event delivery", "## Verified facts", "## Hypotheses"):
        assert section in md
    assert "CONFIRMED" in md and "SENT, NOT CONFIRMED" in md and "LOCAL-ONLY" in md


def test_report_never_claims_delivery_when_nothing_was_sent():
    inv = investigate(index, graph, "Which functions call login()?")
    md = to_markdown(Report(inv, index, graph, deliveries=[], dnotifier_state="DNotifier disabled"))
    assert "No events were sent" in md and "CONFIRMED" not in md

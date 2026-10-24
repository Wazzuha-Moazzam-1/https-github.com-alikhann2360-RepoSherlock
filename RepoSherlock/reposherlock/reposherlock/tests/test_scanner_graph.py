from pathlib import Path

import pytest

from graph_builder import build_call_graph, callers_of, find_cycles, find_paths
from scanner import scan_repository

SAMPLE = Path(__file__).resolve().parent.parent / "sample_project"


@pytest.fixture(scope="module")
def index():
    return scan_repository(SAMPLE)


@pytest.fixture(scope="module")
def graph(index):
    return build_call_graph(index)


def fid(index, name):
    return next(f.id for f in index.real_functions() if f.name == name)


def test_functions_and_classes_found(index):
    names = {f.name for f in index.real_functions()}
    assert {"login", "validate_credentials", "connect_database", "main", "create_session"} <= names
    assert [c.name for c in index.classes] == ["DatabaseError"]


def test_line_numbers_match_source(index):
    for name in ("login", "validate_credentials", "connect_database"):
        f = next(x for x in index.real_functions() if x.name == name)
        src = index.lines[f.file]
        assert src[f.start_line - 1].lstrip().startswith(f"def {name}(")
        assert f.end_line >= f.start_line


def test_calls_extracted_with_lines(index):
    login = next(f for f in index.real_functions() if f.name == "login")
    called = {c.name: c.line for c in login.calls}
    assert {"connect_database", "validate_credentials", "create_session"} <= set(called)
    assert index.lines["auth.py"][called["create_session"] - 1].strip().endswith("create_session(username)")


def test_call_graph_edges(index, graph):
    login, validate = fid(index, "login"), fid(index, "validate_credentials")
    assert graph.has_edge(login, validate)
    assert login in callers_of(graph, validate)


def test_mutual_recursion_cycle_is_handled(index, graph):
    cycles = find_cycles(graph)
    assert any({index.functions[n].name for n in c} == {"connect_database", "retry_connect"} for c in cycles)
    paths = find_paths(graph, fid(index, "main"), fid(index, "connect_database"))
    assert paths and all(len(set(p)) == len(p) for p in paths)   # simple paths: no revisits


def test_self_recursion(tmp_path):
    (tmp_path / "r.py").write_text("def f(n):\n    return f(n - 1) if n else 0\n", encoding="utf-8")
    idx = scan_repository(tmp_path)
    g = build_call_graph(idx)
    assert any(len(c) == 1 for c in find_cycles(g))


def test_skips_irrelevant_dirs(tmp_path):
    for d in (".venv", "node_modules", ".git", "venv"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "x.py").write_text("def hidden(): pass\n", encoding="utf-8")
    (tmp_path / "ok.py").write_text("def visible(): pass\n", encoding="utf-8")
    idx = scan_repository(tmp_path)
    assert idx.files == ["ok.py"]


def test_malformed_file_does_not_crash(tmp_path):
    (tmp_path / "bad.py").write_text("def broken(:\n    pass\n\ndef fine():\n    return 1\n", encoding="utf-8")
    (tmp_path / "bin.py").write_bytes(b"\xff\xfe\x00 def x(): pass")
    idx = scan_repository(tmp_path)
    assert any("syntax errors" in w for w in idx.warnings)
    assert "fine" in {f.name for f in idx.real_functions()}


def test_never_executes_repository_code(tmp_path):
    marker = tmp_path / "pwned.txt"
    (tmp_path / "evil.py").write_text(
        f"open(r'{marker}', 'w').write('x')\ndef f():\n    open(r'{marker}', 'w').write('x')\n", encoding="utf-8")
    scan_repository(tmp_path)
    assert not marker.exists()


def test_methods_get_class_qualnames(tmp_path):
    (tmp_path / "c.py").write_text(
        "class A:\n    def run(self):\n        self.step()\n    def step(self):\n        pass\n", encoding="utf-8")
    idx = scan_repository(tmp_path)
    g = build_call_graph(idx)
    assert g.has_edge("c.py::A.run", "c.py::A.step")

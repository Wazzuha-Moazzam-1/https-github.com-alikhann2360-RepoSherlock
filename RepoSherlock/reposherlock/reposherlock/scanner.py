"""Repository scanner: finds Python files and extracts functions, classes and calls
with Tree-sitter. Read-only: it parses text and never imports or executes anything.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

SKIP_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", ".env", "node_modules",
    "__pycache__", ".tox", ".nox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "build", "dist", "site-packages", ".idea", ".vscode", ".eggs",
}
MAX_FILE_BYTES = 1_000_000  # skip huge/generated files

_PY_LANGUAGE = Language(tspython.language())
_parser = Parser(_PY_LANGUAGE)

MODULE_NAME = "<module>"


@dataclass
class CallSite:
    name: str          # last identifier, e.g. "login" for auth.login(...)
    text: str          # callee expression as written, e.g. "auth.login"
    line: int          # 1-based


@dataclass
class FunctionInfo:
    id: str            # "relative/path.py::Class.method"
    name: str
    qualname: str
    file: str          # path relative to the repo root, forward slashes
    start_line: int
    end_line: int
    kind: str          # "function" | "method" | "module"
    class_name: str | None = None
    docstring: str = ""
    calls: list[CallSite] = field(default_factory=list)


@dataclass
class ClassInfo:
    name: str
    file: str
    start_line: int
    end_line: int
    methods: list[str] = field(default_factory=list)


@dataclass
class RepoIndex:
    root: Path
    files: list[str] = field(default_factory=list)
    functions: dict[str, FunctionInfo] = field(default_factory=dict)
    classes: list[ClassInfo] = field(default_factory=list)
    lines: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def real_functions(self) -> list[FunctionInfo]:
        return [f for f in self.functions.values() if f.kind != "module"]

    def get_source(self, func: FunctionInfo, max_lines: int = 40) -> tuple[int, list[str], bool]:
        """Return (first_line_number, lines, truncated) for a function body."""
        all_lines = self.lines.get(func.file, [])
        start, end = func.start_line, func.end_line
        chunk = all_lines[start - 1:end]
        truncated = len(chunk) > max_lines
        return start, chunk[:max_lines], truncated


def find_python_files(root: Path) -> list[Path]:
    found: list[Path] = []
    # followlinks=False (default): never wander outside the repo through symlinks
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info")
        )
        for name in sorted(filenames):
            if name.endswith(".py"):
                found.append(Path(dirpath) / name)
    return found


def _text(node) -> str:
    return node.text.decode("utf-8", errors="replace") if node is not None else ""


def _docstring(body_node) -> str:
    if body_node is None or not body_node.named_children:
        return ""
    first = body_node.named_children[0]
    if first.type == "expression_statement" and first.named_children:
        s = first.named_children[0]
        if s.type == "string":
            raw = _text(s).strip()
            return raw.strip("\"'").strip()[:300]
    return ""


def _callee(call_node) -> CallSite | None:
    fn = call_node.child_by_field_name("function")
    if fn is None:
        return None
    text = _text(fn)
    if fn.type == "identifier":
        name = text
    elif fn.type == "attribute":
        name = _text(fn.child_by_field_name("attribute"))
    else:
        return None  # e.g. calling the result of another call; skip
    if not name:
        return None
    return CallSite(name=name, text=text[:80], line=call_node.start_point[0] + 1)


def _walk(node, rel: str, index: RepoIndex, current: FunctionInfo, class_stack: list[str],
          module_info: FunctionInfo) -> None:
    """Depth-first walk. `current` is the innermost function that owns calls we find."""
    t = node.type
    if t == "class_definition":
        name = _text(node.child_by_field_name("name"))
        cls = ClassInfo(name=name, file=rel, start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1)
        index.classes.append(cls)
        class_stack.append(name)
        for child in node.children:
            _walk(child, rel, index, current, class_stack, module_info)
        class_stack.pop()
        return
    if t == "function_definition":
        name = _text(node.child_by_field_name("name"))
        in_class = class_stack[-1] if class_stack and current is module_info else None
        qual = f"{in_class}.{name}" if in_class else name
        if current is not module_info and current.kind != "module":
            qual = f"{current.qualname}.<locals>.{name}"
        fid = f"{rel}::{qual}"
        if fid in index.functions:  # redefinition in same file: keep both, disambiguate
            fid = f"{fid}@{node.start_point[0] + 1}"
        info = FunctionInfo(
            id=fid, name=name, qualname=qual, file=rel,
            start_line=node.start_point[0] + 1, end_line=node.end_point[0] + 1,
            kind="method" if in_class else "function", class_name=in_class,
            docstring=_docstring(node.child_by_field_name("body")),
        )
        index.functions[fid] = info
        if in_class:
            for c in reversed(index.classes):
                if c.name == in_class and c.file == rel:
                    c.methods.append(name)
                    break
        # Decorators/defaults are evaluated by the enclosing scope, but we keep it simple:
        # everything inside the def belongs to this function.
        for child in node.children:
            _walk(child, rel, index, info, class_stack, module_info)
        return
    if t == "call":
        site = _callee(node)
        if site is not None:
            current.calls.append(site)
    for child in node.children:
        _walk(child, rel, index, current, class_stack, module_info)


def scan_file(path: Path, root: Path, index: RepoIndex) -> None:
    rel = path.relative_to(root).as_posix()
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            index.warnings.append(f"{rel}: skipped (larger than {MAX_FILE_BYTES // 1000} KB)")
            return
        data = path.read_bytes()
    except OSError as exc:
        index.warnings.append(f"{rel}: could not be read ({exc.__class__.__name__})")
        return

    try:
        tree = _parser.parse(data)
    except Exception as exc:  # defensive: parser should never raise, but never crash the scan
        index.warnings.append(f"{rel}: parse failed ({exc.__class__.__name__})")
        return

    index.files.append(rel)
    index.lines[rel] = data.decode("utf-8", errors="replace").splitlines()
    if tree.root_node.has_error:
        index.warnings.append(f"{rel}: contains syntax errors; results for this file may be incomplete")

    module_info = FunctionInfo(
        id=f"{rel}::{MODULE_NAME}", name=MODULE_NAME, qualname=MODULE_NAME, file=rel,
        start_line=1, end_line=max(1, len(index.lines[rel])), kind="module",
    )
    _walk(tree.root_node, rel, index, module_info, [], module_info)
    if module_info.calls:
        index.functions[module_info.id] = module_info


def scan_repository(root: str | Path) -> RepoIndex:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    index = RepoIndex(root=root)
    for path in find_python_files(root):
        scan_file(path, root, index)
    return index

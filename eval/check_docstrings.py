"""Docstring sweep: every public function/class/method in harness/ needs one.

Walk ``harness/`` with stdlib ``ast`` only (no imports, no deps). A definition
counts as public when its name does not start with a single underscore;
dunder methods (``__init__`` etc.) count as public protocol and require
docstrings too. Nested functions and methods are checked; module docstrings
are not required (so empty ``__init__.py`` files pass).

Exit 0 when clean; exit 1 listing every undocumented public definition as
``path:lineno: qualified.name``.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "harness"


def _is_public(name: str) -> bool:
    """Public unless single-leading-underscore private (dunders are public)."""
    return not (name.startswith("_") and not name.startswith("__"))


def _walk(tree: ast.Module) -> list[tuple[str, int]]:
    """Collect (qualified name, lineno) of public defs missing docstrings."""
    missing: list[tuple[str, int]] = []

    def visit(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qual = f"{scope}.{child.name}" if scope else child.name
                if _is_public(child.name) and ast.get_docstring(child) is None:
                    missing.append((qual, child.lineno))
                visit(child, qual)
            else:
                visit(child, scope)

    visit(tree, "")
    return missing


def sweep(target: Path = TARGET) -> list[str]:
    """Return sorted ``path:lineno: qualname`` lines for undocumented defs."""
    hits: list[str] = []
    for path in sorted(target.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            hits.append(f"{path.relative_to(ROOT)}:0: <unparseable: {exc}>")
            continue
        for qual, lineno in _walk(tree):
            hits.append(f"{path.relative_to(ROOT)}:{lineno}: {qual}")
    return sorted(hits)


def main(argv: list[str] | None = None) -> int:
    """CLI: sweep harness/ and exit 1 listing gaps, 0 when clean."""
    _ = argv
    hits = sweep()
    if hits:
        print(f"FAIL docstring sweep: {len(hits)} undocumented public definition(s):")
        for h in hits:
            print(f"  {h}")
        return 1
    print("PASS docstring sweep: all public harness definitions documented.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

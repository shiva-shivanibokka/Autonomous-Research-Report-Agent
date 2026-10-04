"""
`httpx2` ships with anthropic 1.x but not with the older SDK that
requirements.txt can resolve to, so nothing under eval_sop/ may import it
unconditionally: one such import aborted `pytest eval_sop/tests -q` at
collection, which is the command RESULTS.md section 8 tells people to run.
"""

import ast
from pathlib import Path

EVAL_SOP = Path(__file__).resolve().parents[1]
OPTIONAL = {"httpx2"}


def _unguarded(tree: ast.Module) -> set[str]:
    """Modules in OPTIONAL imported outside a try/except at module level."""
    guarded = {n for node in ast.walk(tree) if isinstance(node, ast.Try)
               for n in _imported(node)}
    return (_imported(tree) & OPTIONAL) - guarded


def _imported(node) -> set[str]:
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Import):
            out |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module.split(".")[0])
    return out


def test_no_unguarded_optional_imports():
    offenders = {}
    for f in EVAL_SOP.rglob("*.py"):
        bad = _unguarded(ast.parse(f.read_text(encoding="utf-8")))
        if bad:
            offenders[str(f.relative_to(EVAL_SOP))] = sorted(bad)
    assert offenders == {}


def test_the_guard_catches_the_pattern_it_is_meant_to():
    assert _unguarded(ast.parse("import httpx2\n")) == {"httpx2"}
    assert _unguarded(ast.parse("try:\n    import httpx2\nexcept ImportError:\n    pass\n")) == set()

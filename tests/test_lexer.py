"""
Regression tests for app.compiler.lex: reserved keywords must only match
as whole tokens, not as a prefix of a longer identifier.

Background: PLY tries function-based token rules (t_SELECT, t_FROM, ...)
in the order they are defined in the module, matching with re.match
(anchored only at the start of the current position, not the end). Every
reserved-word rule in lex.py is defined before t_SIMPLE_COLNAME, so for an
identifier that merely *starts with* a reserved keyword - e.g. a column
named "order_id", "description", or "joint_account" - the keyword rule
used to match just its own literal letters and win, splitting the rest of
the identifier off into a separate, unrelated token. A `\\b` word-boundary
suffix on each keyword pattern fixes this.
"""

import pytest

# Deliberately reuse the app's own singleton lexer (app.compiler.lexer)
# rather than building a fresh one with ply.lex.lex(module=...): PLY tracks
# the most recently built lexer as the implicit default that
# app.compiler.parser.parse() falls back to when called without an
# explicit lexer= argument (see app/etl/controllers.py). Building extra
# lexer instances here would silently hijack that global and break every
# other test in the suite that compiles a query end-to-end.
from app.compiler import lexer
from app.core.result_monad import Success
from app.etl.controllers import compile_to_python, execute_python_code


def _tokenize(text):
    lexer.input(text)
    tokens = []
    while True:
        tok = lexer.token()
        if not tok:
            break
        tokens.append((tok.type, tok.value))
    return tokens


@pytest.mark.parametrize(
    "identifier",
    [
        "selected",
        "fromage",
        "ordering",
        "oncology",
        "ascending",
        "description",
        "limiter",
        "joint",
        "setting",
        "update_time",
        "deleted_at",
        "order_id",
        "joint_account",
        "insertion",
        "valuesets",
        "leftover",
        "rightful",
        "fullname",
        "outerspace",
        "intox",
        "byte",
        "asceticism",
        "distinctive",
        "groupon",
        "wherein",
        "likewise",
        "notification",
        "android",  # starts with "and"
        "ornament",  # starts with "or"
        "tailor",
        "innermost",
    ],
)
def test_identifier_with_keyword_prefix_is_a_single_token(identifier):
    tokens = _tokenize(identifier)
    assert tokens == [("SIMPLE_COLNAME", identifier)], (
        f"{identifier!r} was split into multiple tokens instead of being "
        f"recognized as a single identifier: {tokens}"
    )


@pytest.mark.parametrize(
    "text,expected_type",
    [
        ("select", "SELECT"),
        ("distinct", "DISTINCT"),
        ("from", "FROM"),
        ("into", "INTO"),
        ("group", "GROUP"),
        ("order", "ORDER"),
        ("by", "BY"),
        ("where", "WHERE"),
        ("like", "LIKE"),
        ("not", "NOT"),
        ("and", "AND"),
        ("or", "OR"),
        ("insert", "INSERT"),
        ("values", "VALUES"),
        ("update", "UPDATE"),
        ("set", "SET"),
        ("delete", "DELETE"),
        ("desc", "DESC"),
        ("asc", "ASC"),
        ("limit", "LIMIT"),
        ("tail", "TAIL"),
        ("as", "AS"),
        ("join", "JOIN"),
        ("inner", "INNER"),
        ("left", "LEFT"),
        ("right", "RIGHT"),
        ("full", "FULL"),
        ("outer", "OUTER"),
        ("on", "ON"),
    ],
)
def test_real_keyword_still_recognized_on_its_own(text, expected_type):
    tokens = _tokenize(text)
    assert tokens == [(expected_type, text)]


def test_keyword_followed_by_punctuation_still_recognized():
    # A word boundary also holds between a letter and punctuation/EOF, so
    # this must keep working after the \b fix.
    assert _tokenize("select;") == [("SELECT", "select"), ("SIMICOLON", ";")]
    assert _tokenize("select(") == [("SELECT", "select"), ("LPAREN", "(")]


def test_query_with_keyword_prefixed_column_name_compiles_and_runs(tmp_path):
    # End-to-end regression: before the fix, selecting a column literally
    # named "order_id" would mis-tokenize into ORDER + SIMPLE_COLNAME("_id")
    # and fail to parse at all.
    import csv

    csv_path = tmp_path / "orders.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["order_id", "description", "update_time"])
        writer.writerow([1, "widget", "2026-01-01"])
        writer.writerow([2, "gadget", "2026-01-02"])

    query = f"SELECT order_id, description FROM {{csv:{csv_path}}};"
    compile_result = compile_to_python(query)
    assert isinstance(compile_result, Success), (
        f"compile failed: {compile_result.unwrap_error()}"
    )
    exec_result = execute_python_code(compile_result.unwrap())
    assert isinstance(exec_result, Success), (
        f"execution failed: {exec_result.unwrap_error().message}"
    )
    df = exec_result.unwrap()
    assert list(df.columns) == ["order_id", "description"]
    assert len(df) == 2

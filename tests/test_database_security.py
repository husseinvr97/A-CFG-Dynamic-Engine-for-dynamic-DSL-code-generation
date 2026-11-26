"""
Regression tests for the SQL-injection fix in app/etl/data/local/database.py.

Before the fix, IDatabase.extract() built its query with an f-string:
    pd.read_sql(f"select * from {self.table_name};", self.engine)
Since `table_name` comes straight from the `{sqlite:...}` / `{mssql:...}`
segment of the raw DSL query text, this let a query smuggle arbitrary SQL
in through the "table name" slot, e.g.:
    SELECT * FROM {sqlite:demo.db|users UNION SELECT id, token FROM secrets --};
which returned rows from an unrelated `secrets` table.

These tests reproduce that exact exploit against a real SQLite database and
assert it is now rejected, and that ordinary single- and schema-qualified
table names still work end to end.

Run with:  pytest tests/test_database_security.py
"""

import sqlite3

import pandas as pd
import pytest

from app.core.result_monad import Success
from app.etl.controllers import compile_to_python, execute_python_code
from app.etl.data.local.database import SQLITEDatabase


def _make_db(path):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob')")
    conn.execute("CREATE TABLE secrets (id INTEGER, token TEXT)")
    conn.execute("INSERT INTO secrets VALUES (1, 'TOPSECRET-API-KEY')")
    conn.commit()
    conn.close()


def run_query(query):
    """Compile + execute a query, returning the resulting DataFrame."""
    result = compile_to_python(query)
    assert isinstance(result, Success), f"compile failed: {result.unwrap_error()}"
    code = result.unwrap()
    exec_result = execute_python_code(code)
    assert isinstance(exec_result, Success), (
        f"execution failed: {exec_result.unwrap_error().message}\n"
        f"--- generated code ---\n{code}"
    )
    return exec_result.unwrap()


# ---------------------------------------------------------------------------
# Unit-level: the extractor/loader class itself
# ---------------------------------------------------------------------------

def test_union_injection_in_table_name_is_rejected(tmp_path):
    db_path = str(tmp_path / "demo.db")
    _make_db(db_path)

    malicious_table = "users UNION SELECT id, token FROM secrets --"
    with pytest.raises(ValueError):
        SQLITEDatabase(f"{db_path}|{malicious_table}")


def test_other_injection_shapes_are_rejected(tmp_path):
    db_path = str(tmp_path / "demo.db")
    _make_db(db_path)

    for malicious_table in [
        "users; DROP TABLE users; --",
        "users'--",
        'users"--',
        "users/*comment*/UNION/**/SELECT/**/1",
        "",
        "   ",
        "1users",  # starts with a digit
    ]:
        with pytest.raises(ValueError):
            SQLITEDatabase(f"{db_path}|{malicious_table}")


def test_legitimate_table_name_still_extracts(tmp_path):
    db_path = str(tmp_path / "demo.db")
    _make_db(db_path)

    extractor = SQLITEDatabase(f"{db_path}|users")
    df = extractor.extract()
    assert sorted(df["name"]) == ["Alice", "Bob"]


def test_schema_qualified_table_name_is_allowed(tmp_path):
    # main.users is a valid SQLite schema-qualified name; this exercises the
    # per-segment quoting path without needing a real second schema.
    db_path = str(tmp_path / "demo.db")
    _make_db(db_path)

    extractor = SQLITEDatabase(f"{db_path}|main.users")
    df = extractor.extract()
    assert sorted(df["name"]) == ["Alice", "Bob"]


# ---------------------------------------------------------------------------
# End-to-end: the exact exploit shape from the original finding, through the
# real DSL compiler/executor, the way a user would actually trigger it.
# ---------------------------------------------------------------------------

def test_end_to_end_union_injection_query_is_blocked_not_executed(tmp_path):
    db_path = str(tmp_path / "demo.db")
    _make_db(db_path)
    db_path_sql = db_path.replace("\\", "/")

    query = (
        "SELECT * FROM {sqlite:"
        f"{db_path_sql}|users UNION SELECT id, token FROM secrets --"
        "};"
    )

    compile_result = compile_to_python(query)
    assert isinstance(compile_result, Success), (
        f"compile failed unexpectedly: {compile_result.unwrap_error()}"
    )
    code = compile_result.unwrap()

    exec_result = execute_python_code(code)
    # The malicious query must NOT succeed and must NOT leak `secrets` rows.
    assert not isinstance(exec_result, Success), (
        "SQL injection succeeded instead of being rejected: "
        f"{exec_result.unwrap() if isinstance(exec_result, Success) else ''}"
    )
    assert "Invalid table name" in exec_result.unwrap_error().message


def test_end_to_end_legitimate_sqlite_query_still_works(tmp_path):
    db_path = str(tmp_path / "demo.db")
    _make_db(db_path)
    db_path_sql = db_path.replace("\\", "/")

    query = "SELECT * FROM {sqlite:" + db_path_sql + "|users};"
    df = run_query(query)
    assert sorted(df["name"]) == ["Alice", "Bob"]

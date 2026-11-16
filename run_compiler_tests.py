#!/usr/bin/env python3
"""
Standalone regression test for the QueryFlow SQL-like DSL compiler
(app/compiler + app/etl). Runs headlessly - no GUI / Tkinter required.

Covers:
  - plain SELECT (all columns, specific columns, WHERE, ORDER BY, LIMIT/TAIL)
  - DISTINCT
  - GROUP BY + aggregation functions
  - INSERT
  - JOIN (INNER / LEFT), including:
      - SELECT * through a join
      - alias-qualified columns with no name collision
      - alias-qualified columns WITH a name collision (suffix resolution)
      - alias-qualified columns in WHERE
  - error handling: malformed SQL should fail to compile, not crash

Usage:
    python run_compiler_tests.py
Exit code is 0 if every test passes, 1 otherwise - safe to use in CI
or a pre-commit hook.
"""

import csv
import os
import shutil
import sys
import tempfile
import traceback

# Make sure "app" is importable regardless of where this script is invoked from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.result_monad import Success  # noqa: E402
from app.etl.controllers import compile_to_python, execute_python_code  # noqa: E402


# ---------------------------------------------------------------------------
# Test data setup
# ---------------------------------------------------------------------------

def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def make_test_datasets(tmp_dir):
    """Create small, deterministic CSVs so this script never depends on
    files that may or may not exist in the repo (e.g. a.csv)."""
    paths = {}

    paths["areas"] = os.path.join(tmp_dir, "areas.csv")
    _write_csv(
        paths["areas"],
        ["name", "longitude", "latitude"],
        [
            ["Wadi Araba", 32.446042, 29.12096],
            ["Ras Ghareb", 32.928549, 28.19849],
            ["Linz", 14.286110, 48.30639],
        ],
    )

    # No column-name collision with areas.csv (apart from the join key).
    paths["weather"] = os.path.join(tmp_dir, "weather.csv")
    _write_csv(
        paths["weather"],
        ["name", "temperature", "humidity"],
        [
            ["Wadi Araba", 34.5, 22],
            ["Ras Ghareb", 31.2, 30],
            ["Linz", 18.9, 55],
        ],
    )

    # Deliberately collides on "longitude"/"latitude" with areas.csv, to
    # exercise pandas' merge-suffix behaviour.
    paths["weather_collide"] = os.path.join(tmp_dir, "weather_collide.csv")
    _write_csv(
        paths["weather_collide"],
        ["name", "longitude", "latitude", "temperature"],
        [
            ["Wadi Araba", 32.9, 29.5, 34.5],
            ["Ras Ghareb", 32.8, 28.5, 31.2],
            ["Linz", 14.9, 48.5, 18.9],
        ],
    )

    # A row in weather_partial.csv that areas.csv has no match for, to
    # exercise LEFT JOIN producing a NULL/NaN row.
    paths["weather_partial"] = os.path.join(tmp_dir, "weather_partial.csv")
    _write_csv(
        paths["weather_partial"],
        ["name", "temperature"],
        [
            ["Wadi Araba", 34.5],
            # Ras Ghareb and Linz deliberately missing
        ],
    )

    paths["people"] = os.path.join(tmp_dir, "people.csv")
    _write_csv(
        paths["people"],
        ["id", "name", "profession", "cost"],
        [
            [1, "Alice", "doctor", 5200],
            [2, "Bob", "doctor", 4800],
            [3, "Cara", "developer", 6100],
            [4, "Dan", "developer", 5900],
            [5, "Eve", "worker", 3000],
        ],
    )

    return paths


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

class TestCase:
    def __init__(self, name, query, check):
        self.name = name
        self.query = query
        self.check = check  # callable(df) -> None, raises AssertionError on failure


class ExpectCompileFailure:
    def __init__(self, name, query):
        self.name = name
        self.query = query


def run_case(case):
    result = compile_to_python(case.query)
    if not isinstance(result, Success):
        return False, f"compile failed: {result.unwrap_error()}"

    code = result.unwrap()
    exec_result = execute_python_code(code)
    if not isinstance(exec_result, Success):
        return False, f"execution failed: {exec_result.unwrap_error().message}"

    df = exec_result.unwrap()
    try:
        case.check(df)
    except AssertionError as e:
        return False, f"result check failed: {e}\n--- generated code ---\n{code}"
    except Exception as e:
        return False, f"result check raised {type(e).__name__}: {e}"
    return True, None


def run_expect_failure(case):
    result = compile_to_python(case.query)
    if isinstance(result, Success):
        # It compiled - see if it at least fails at execution instead.
        exec_result = execute_python_code(result.unwrap())
        if isinstance(exec_result, Success):
            return False, "expected this query to fail (compile or execute), but it succeeded"
    return True, None


def build_test_suite(paths):
    tests = []

    tests.append(TestCase(
        "SELECT * (plain)",
        f"SELECT * FROM {{csv:{paths['areas']}}} LIMIT 5;",
        lambda df: (
            _assert_eq(list(df.columns), ["name", "longitude", "latitude"]),
            _assert_eq(len(df), 3),
        ),
    ))

    tests.append(TestCase(
        "SELECT specific columns",
        f"SELECT name, longitude FROM {{csv:{paths['areas']}}};",
        lambda df: _assert_eq(list(df.columns), ["name", "longitude"]),
    ))

    tests.append(TestCase(
        "WHERE + ORDER BY + LIMIT",
        f"SELECT name, cost FROM {{csv:{paths['people']}}} WHERE cost > 5000 "
        f"ORDER BY cost DESC LIMIT 2;",
        lambda df: (
            _assert_eq(len(df), 2),
            _assert_eq(df.iloc[0]["name"], "Cara"),
        ),
    ))

    tests.append(TestCase(
        "DISTINCT",
        f"SELECT DISTINCT profession FROM {{csv:{paths['people']}}};",
        lambda df: _assert_eq(sorted(df["profession"].tolist()),
                               ["developer", "doctor", "worker"]),
    ))

    tests.append(TestCase(
        "GROUP BY + aggregation",
        f"SELECT profession, mean(cost) FROM {{csv:{paths['people']}}} GROUP BY profession;",
        lambda df: _assert_eq(len(df), 3),
    ))

    tests.append(TestCase(
        "INNER JOIN, no column collision",
        f"SELECT A.name, A.longitude, B.temperature "
        f"FROM {{csv:{paths['areas']}}} AS A "
        f"JOIN {{csv:{paths['weather']}}} AS B ON A.name = B.name;",
        lambda df: (
            _assert_eq(list(df.columns), ["name", "longitude", "temperature"]),
            _assert_eq(len(df), 3),
            _assert_eq(float(df.loc[df["name"] == "Linz", "temperature"].iloc[0]), 18.9),
        ),
    ))

    tests.append(TestCase(
        "SELECT * through a JOIN",
        f"SELECT * FROM {{csv:{paths['areas']}}} AS A "
        f"JOIN {{csv:{paths['weather']}}} AS B ON A.name = B.name;",
        lambda df: _assert_eq(len(df), 3),
    ))

    tests.append(TestCase(
        "LEFT JOIN + WHERE on an alias-qualified column",
        f"SELECT A.name, B.temperature "
        f"FROM {{csv:{paths['areas']}}} AS A "
        f"LEFT JOIN {{csv:{paths['weather']}}} AS B ON A.name = B.name "
        f"WHERE B.temperature > 20;",
        lambda df: (
            _assert_eq(len(df), 2),
            _assert_eq(sorted(df["name"].tolist()), ["Ras Ghareb", "Wadi Araba"]),
        ),
    ))

    tests.append(TestCase(
        "LEFT JOIN with an unmatched row (NaN expected)",
        f"SELECT A.name, B.temperature "
        f"FROM {{csv:{paths['areas']}}} AS A "
        f"LEFT JOIN {{csv:{paths['weather_partial']}}} AS B ON A.name = B.name;",
        lambda df: (
            _assert_eq(len(df), 3),
            _assert_true(df.loc[df["name"] == "Linz", "temperature"].isna().all(),
                         "expected Linz's temperature to be NaN (no match in weather_partial.csv)"),
        ),
    ))

    tests.append(TestCase(
        "JOIN with a genuine column-name collision (suffix resolution)",
        f"SELECT A.name, B.longitude, B.temperature "
        f"FROM {{csv:{paths['areas']}}} AS A "
        f"JOIN {{csv:{paths['weather_collide']}}} AS B ON A.name = B.name;",
        lambda df: _assert_eq(len(df), 3),
    ))

    return tests


def run_insert_test(tmp_dir):
    """
    INSERT doesn't return a DataFrame the way SELECT does - it writes a
    file - so it needs its own check: compile, execute, then read the file
    back and verify both the values and the on-disk shape are correct.
    """
    import pandas as pd

    target = os.path.join(tmp_dir, "insert_target.csv")
    # This DSL's string literals are double-quoted only (see t_STRING in
    # app/compiler/lex.py) - single quotes are not valid syntax here.
    query = f'INSERT INTO {{csv:{target}}} (name, age) VALUES ("Alice", 30);'

    result = compile_to_python(query)
    if not isinstance(result, Success):
        return False, f"compile failed: {result.unwrap_error()}"

    exec_result = execute_python_code(result.unwrap())
    if not isinstance(exec_result, Success):
        return False, f"execution failed: {exec_result.unwrap_error().message}"

    if not os.path.exists(target):
        return False, f"INSERT ran but {target} was never created"

    written = pd.read_csv(target)
    try:
        _assert_eq(list(written.columns), ["name", "age"])
        _assert_eq(written.iloc[0]["name"], "Alice")  # not '"Alice"' with quotes
        _assert_eq(int(written.iloc[0]["age"]), 30)
    except AssertionError as e:
        return False, f"{e}\n--- written file contents ---\n{written}"

    return True, None


def _assert_eq(actual, expected):
    assert actual == expected, f"expected {expected!r}, got {actual!r}"


def _assert_true(value, message):
    assert bool(value), message


def build_failure_cases(paths):
    return [
        ExpectCompileFailure(
            "malformed JOIN (ON clause missing a comparison)",
            f"SELECT * FROM {{csv:{paths['areas']}}} AS A "
            f"JOIN {{csv:{paths['weather']}}} ON A.name;",
        ),
        ExpectCompileFailure(
            "garbage input",
            "THIS IS NOT SQL AT ALL;",
        ),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tmp_dir = tempfile.mkdtemp(prefix="queryflow_test_")
    passed, failed = 0, 0
    failures = []

    try:
        paths = make_test_datasets(tmp_dir)

        print("=" * 70)
        print("QueryFlow compiler/ETL regression tests")
        print("=" * 70)

        for case in build_test_suite(paths):
            ok, detail = run_case(case)
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] {case.name}")
            if ok:
                passed += 1
            else:
                failed += 1
                failures.append((case.name, detail))

        ok, detail = run_insert_test(tmp_dir)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] INSERT writes correct, unquoted values with no stray index column")
        if ok:
            passed += 1
        else:
            failed += 1
            failures.append(("INSERT round-trip", detail))

        for case in build_failure_cases(paths):
            ok, detail = run_expect_failure(case)
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] {case.name} (expected to fail)")
            if ok:
                passed += 1
            else:
                failed += 1
                failures.append((case.name, detail))

        print("=" * 70)
        print(f"{passed} passed, {failed} failed")

        if failures:
            print()
            print("Details of failures:")
            for name, detail in failures:
                print(f"\n--- {name} ---")
                print(detail)

        return 0 if failed == 0 else 1

    except Exception:
        print("UNEXPECTED ERROR while running the test suite:")
        traceback.print_exc()
        return 1

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

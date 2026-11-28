"""
Regression tests for the global-mutable-transformed_data race condition.

Background: execute_python_code() used to run `exec(python_code)` with no
explicit globals dict. The generated code's `transformed_data = ...` line
therefore ran against the calling function's own namespace, and the real
result was smuggled out through a *module-level* global
(app.etl.core.transformed_data) that transform_select()/join() set as a
side effect. execute_python_code() then read that same shared global back
via `from app.etl.core import transformed_data`.

That global is one mailbox shared by every call in the whole process. Two
queries executing concurrently - e.g. two background threads from the GUI
(see the Execute-button threading fix), two tabs, or just two quick
successive clicks - could silently overwrite each other's result: whichever
thread's transform_select() happened to run last "won" the global, and any
other thread still mid-flight read back the wrong data under its own name.

The fix executes each call into its own private `exec_globals` dict and
reads `transformed_data` back from that dict, so there is no longer any
shared mutable state between concurrent calls.
"""

import threading
from unittest import mock

import app.etl as etl_pkg
from app.etl.controllers import compile_to_python, execute_python_code


def test_concurrent_queries_do_not_leak_results_into_each_other(tmp_path):
    # Two distinct, easily-told-apart datasets.
    path_a = tmp_path / "a.csv"
    path_b = tmp_path / "b.csv"
    path_a.write_text("name,cost\nAAAAA,111\n")
    path_b.write_text("name,cost\nBBBBB,222\n")

    query_a = f"SELECT * FROM {{csv:{path_a}}};"
    query_b = f"SELECT * FROM {{csv:{path_b}}};"

    real_transform_select = etl_pkg.transform_select

    # Force the exact interleaving the real bug depends on, deterministically,
    # instead of hoping for an unlucky GIL switch: pause thread A right after
    # its transform_select() call has completed (and, under the old code,
    # would have just written the shared global) but before
    # execute_python_code() has read the result back out. Let thread B run to
    # completion - including its own read - while A is paused, then release A.
    a_reached_pause = threading.Event()
    b_finished = threading.Event()

    def patched_transform_select(data, criteria):
        result = real_transform_select(data, criteria)
        if threading.current_thread().name == "QueryA":
            a_reached_pause.set()
            assert b_finished.wait(timeout=5), "thread B did not finish in time"
        return result

    results = {}

    def run_query(name, query):
        threading.current_thread().name = name
        compiled = compile_to_python(query).unwrap()
        results[name] = execute_python_code(compiled)

    with mock.patch.object(
        etl_pkg, "transform_select", side_effect=patched_transform_select
    ):
        t_a = threading.Thread(target=run_query, args=("QueryA", query_a), name="QueryA")
        t_a.start()
        assert a_reached_pause.wait(timeout=5), "thread A never reached the pause point"

        t_b = threading.Thread(target=run_query, args=("QueryB", query_b), name="QueryB")
        t_b.start()
        t_b.join(timeout=5)
        b_finished.set()

        t_a.join(timeout=5)

    result_a = results["QueryA"].unwrap()
    result_b = results["QueryB"].unwrap()

    assert result_a["name"].tolist() == ["AAAAA"], (
        "thread A's query result was contaminated by another thread's "
        f"concurrent execution: got {result_a['name'].tolist()!r}"
    )
    assert result_b["name"].tolist() == ["BBBBB"]


def test_insert_only_query_succeeds_with_no_transformed_data(tmp_path):
    # INSERT compiles to code that calls etl.load(...) and never assigns a
    # `transformed_data` variable at all - that's a legitimate "no result"
    # case, not an error, and execute_python_code() must not crash on it
    # just because no global/local named `transformed_data` exists.
    target = tmp_path / "insert_target.csv"
    query = f'INSERT INTO {{csv:{target}}} (name, age) VALUES ("Alice", 30);'

    compiled = compile_to_python(query).unwrap()
    from app.core.result_monad import Success

    result = execute_python_code(compiled)
    assert isinstance(result, Success), result.unwrap_error()
    assert result.unwrap() is None

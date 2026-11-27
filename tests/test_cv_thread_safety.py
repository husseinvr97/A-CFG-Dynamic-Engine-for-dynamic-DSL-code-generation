"""
Regression tests for the producer/consumer thread-hang fix in app/cv/.

Before the fix, the images pipeline (FolderReader -> DetailsExtractor, driven
by app.cv.operation_main.get_details) and the video pipeline (VideoReader ->
BirdDetector, driven by app.cv.threading_main.read_and_detect) had no error
handling at all:

- If the producer thread raised before pushing its sentinel object onto the
  shared Queue, the consumer thread would block on queue.get() forever.
- If the consumer thread raised before calling lock.release() (a
  threading.Semaphore), the caller would block on lock.acquire() forever.

Since these are reachable from a live DSL query (`SELECT ... FROM
{images:<folder>}` / `{video:<path>}`, see app/etl/data/local/media.py and
app/etl/data/data_factories.py) and that query runs synchronously on the
GUI's Tkinter main thread (see app/gui/vertical_tab_view/content_frame.py),
a single malformed filename in a folder used to freeze the entire
application forever, with no way to recover short of killing the process.

The fix wraps each thread's body in try/except/finally so the sentinel is
always pushed and the lock is always released, and stores any exception on
the thread object so the caller can re-raise it after joining.

These tests prove the fix by actually calling the real functions from a
watchdog thread with a timeout, so a regression back to "hangs forever"
fails the test instead of hanging the test run itself.

Run with:  pytest tests/test_cv_thread_safety.py
"""

import queue
import threading

import pytest

cv2 = pytest.importorskip("cv2", reason="opencv-python not installed")
import numpy as np  # noqa: E402

from app.cv.detector import BirdDetector
from app.cv.operation_main import get_details
from app.cv.threading_main import _BIRDS_CASCADE_PATH, read_and_detect


HANG_TIMEOUT = 10.0  # generous; a regression to "hangs forever" still fails fast


def _run_with_timeout(func, *args, timeout=HANG_TIMEOUT, **kwargs):
    """
    Run `func(*args, **kwargs)` on a background thread and wait up to
    `timeout` seconds for it. Returns (result, exception). If the thread is
    still alive after the timeout, fails the test immediately (rather than
    actually hanging the test suite), because that is exactly the bug this
    file guards against.
    """
    result_box: dict = {}

    def target():
        try:
            result_box["result"] = func(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - want to see it in the test thread
            result_box["exception"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout=timeout)
    if worker.is_alive():
        pytest.fail(
            f"{func.__qualname__} did not return within {timeout}s - "
            "this is exactly the permanent-hang regression this test guards against"
        )
    return result_box.get("result"), result_box.get("exception")


# ---------------------------------------------------------------------------
# Images pipeline: app.cv.operation_main.get_details
# ---------------------------------------------------------------------------

def test_malformed_filename_raises_instead_of_hanging(tmp_path):
    # Filenames the real pipeline expects look like "XXXXhead.a_leg.b_wing.c_tail.d.jpg"
    # (4-char prefix, then 4 underscore-separated "<name>.<pose>" tokens).
    # This one has no underscores at all, so DetailsExtractor's
    # `image.split("_")` unpacking into 4 names raises ValueError.
    (tmp_path / "XXXXbadname.jpg").write_bytes(b"not a real image, just needs the name")

    result, exc = _run_with_timeout(get_details, str(tmp_path))

    assert exc is not None, "expected the malformed filename to raise, not hang or silently succeed"
    assert isinstance(exc, ValueError)


def test_missing_folder_raises_instead_of_hanging(tmp_path):
    missing = str(tmp_path / "does_not_exist")

    result, exc = _run_with_timeout(get_details, missing)

    assert exc is not None, "expected a missing folder to raise, not hang"
    assert isinstance(exc, (FileNotFoundError, NotADirectoryError, OSError))


def test_valid_images_still_extract_correctly(tmp_path):
    # Real convention: 4-char prefix is stripped, then the remaining name is
    # split on "_" into 4 "<label>.<pose>" tokens.
    (tmp_path / "0001head.up_leg.down_wing.open_tail.spread.jpg").write_bytes(b"x")
    (tmp_path / "0002head.down_leg.up_wing.closed_tail.folded.jpg").write_bytes(b"x")

    result, exc = _run_with_timeout(get_details, str(tmp_path))

    assert exc is None, f"did not expect an error for well-formed filenames, got {exc!r}"
    assert len(result) == 2
    assert {row["HEAD"] for row in result} == {"up", "down"}


# ---------------------------------------------------------------------------
# Video pipeline: app.cv.threading_main.read_and_detect
# ---------------------------------------------------------------------------

def test_missing_video_path_does_not_hang_and_reports_zero():
    # cv2.VideoCapture doesn't raise on a bad path; isOpened() is just False
    # and the first read() returns (False, None), so this was never a hang -
    # this test locks in that correct, already-existing behaviour.
    result, exc = _run_with_timeout(read_and_detect, "/no/such/video.mp4")

    assert exc is None
    assert result == {"MaxBirds": 0}


def test_cascade_path_is_portable_and_loads(tmp_path):
    # Guards against a regression back to a hardcoded, machine-specific path.
    assert "F:" not in _BIRDS_CASCADE_PATH
    cascade = cv2.CascadeClassifier(_BIRDS_CASCADE_PATH)
    assert not cascade.empty(), (
        f"expected the real cascade file to load from {_BIRDS_CASCADE_PATH!r}"
    )


def test_real_test_video_runs_end_to_end(tmp_path):
    # app/cv/birds-test.mp4 ships in the repo specifically for this.
    import os

    video_path = os.path.join(os.path.dirname(__file__), "..", "app", "cv", "birds-test.mp4")
    assert os.path.exists(video_path), "expected the repo's own test video to exist"

    result, exc = _run_with_timeout(read_and_detect, video_path, timeout=60.0)

    assert exc is None, f"did not expect the real test video to error, got {exc!r}"
    assert "MaxBirds" in result
    assert isinstance(result["MaxBirds"], int)
    assert result["MaxBirds"] >= 0


# ---------------------------------------------------------------------------
# BirdDetector in isolation: a mid-stream exception (a genuinely corrupt
# frame, not just a bad path) must also release the lock instead of hanging.
# ---------------------------------------------------------------------------

def test_bird_detector_mid_stream_exception_releases_lock():
    q: queue.Queue = queue.Queue()
    sentinel = object()
    lock = threading.Semaphore(0)
    res: dict = {}
    cascade = cv2.CascadeClassifier(_BIRDS_CASCADE_PATH)

    # A real frame first, then a value that breaks cv2.cvtColor mid-stream -
    # simulates a corrupted frame partway through a video, not just a bad path.
    good_frame = np.zeros((10, 10, 3), dtype="uint8")
    q.put(good_frame)
    q.put("not a valid frame")
    q.put(sentinel)

    detector = BirdDetector(queue=q, sentinel=sentinel, lock=lock, res=res, birdsCascade=cascade)
    detector.start()

    acquired = lock.acquire(timeout=HANG_TIMEOUT)
    assert acquired, "lock was never released - BirdDetector hung on a mid-stream error"
    detector.join(timeout=HANG_TIMEOUT)
    assert not detector.is_alive()
    assert detector.error is not None
    assert res.get("MaxBirds") == 0

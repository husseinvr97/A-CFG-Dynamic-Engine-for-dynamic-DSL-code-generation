import os
import threading
from queue import Queue
from typing import Any
import cv2
from app.cv.video_reader import VideoReader
from app.cv.detector import BirdDetector

# Resolve the cascade file relative to this module, not a hardcoded,
# machine-specific Windows path (the original "F:/Collage/4th year/..."
# path only ever existed on the original author's machine and made this
# unusable anywhere else).
_BIRDS_CASCADE_PATH = os.path.join(os.path.dirname(__file__), "birds1.xml")


def read_and_detect(path: str) -> dict[Any, Any]:
    lock = threading.Semaphore(0)
    SENTINEL = object()  # SENTINEL OBJECT
    SHARED_Q = Queue()
    res = dict[Any, Any]()
    birdsCascade = cv2.CascadeClassifier(_BIRDS_CASCADE_PATH)
    if birdsCascade.empty():
        raise FileNotFoundError(
            f"Could not load bird-detection cascade file at {_BIRDS_CASCADE_PATH!r}"
        )
    reader = VideoReader(path=path, sentinel=SENTINEL, queue=SHARED_Q)

    detector = BirdDetector(
        lock=lock, sentinel=SENTINEL, queue=SHARED_Q, res=res, birdsCascade=birdsCascade
    )
    reader.start()
    detector.start()
    lock.acquire()
    # Both threads are guaranteed to have finished by now: the detector only
    # releases `lock` from its own finally block, after it has stopped
    # reading from `SHARED_Q`. These joins just make that explicit and
    # bound (instead of relying on scheduling) before we inspect .error below.
    reader.join()
    detector.join()
    if reader.error is not None:
        raise reader.error
    if detector.error is not None:
        raise detector.error
    return res

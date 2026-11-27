from threading import Thread

import cv2


class VideoReader(Thread):
    def __init__(self, path, queue, sentinel):
        super().__init__(daemon=True)
        self.queue = queue
        self.path = path
        self.sentinel = sentinel
        # Set if run() raises, so the caller (see threading_main.read_and_detect)
        # can re-raise it on its own thread.
        self.error: Exception | None = None

    def run(self):
        cap = None
        try:
            cap = cv2.VideoCapture(self.path)
            ret, frame = cap.read()
            while ret:
                self.queue.put(frame)
                ret, frame = cap.read()
        except Exception as exc:  # noqa: BLE001 - captured to re-raise on the caller's thread
            self.error = exc
        finally:
            if cap is not None:
                cap.release()
            # Always push the sentinel, success or failure, so the consumer
            # thread (BirdDetector) never blocks on queue.get() forever.
            self.queue.put(self.sentinel)

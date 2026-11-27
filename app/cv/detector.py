from threading import Thread
import cv2


class BirdDetector(Thread):
    def __init__(self, queue, sentinel, lock, res, birdsCascade):
        super().__init__(daemon=True)
        self.queue = queue
        self.lock = lock
        self.birdsCascade = birdsCascade
        self.sentinel = sentinel
        self.res = res
        self.MAX_BIRDS = 0
        # Set if run() raises, so the caller (see threading_main.read_and_detect)
        # can re-raise it on its own thread.
        self.error: Exception | None = None

    def run(self):
        try:
            frame = self.queue.get()
            print("Detecting ...")

            while frame is not self.sentinel:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Detect birds in the gray scale image
                birds = self.birdsCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.4,
                    minNeighbors=5,
                    # minSize=(10, 10),
                    maxSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE,
                )
                if len(birds) > self.MAX_BIRDS:
                    self.MAX_BIRDS = len(birds)
                frame = self.queue.get()
        except Exception as exc:  # noqa: BLE001 - captured to re-raise on the caller's thread
            self.error = exc
        finally:
            # Always record a result and release the lock, success or
            # failure, so read_and_detect() never blocks on lock.acquire()
            # forever.
            self.res["MaxBirds"] = self.MAX_BIRDS
            self.lock.release()

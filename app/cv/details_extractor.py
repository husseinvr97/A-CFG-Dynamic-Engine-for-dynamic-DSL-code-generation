from threading import Thread


class DetailsExtractor(Thread):
    def __init__(self, queue, sentinel, lock, res):
        super().__init__(daemon=True)
        self.queue = queue
        self.sentinel = sentinel
        self.lock = lock
        self.res = res
        # Set if extract() raises (e.g. a filename that doesn't match the
        # expected "head_pose.X_leg_pose.Y_..." convention), so the caller
        # (see operation_main.get_details) can re-raise it on its own thread.
        self.error: Exception | None = None

    def extract(self):
        try:
            image = self.queue.get()
            while image is not self.sentinel:
                head_pose, leg_pose, wing_pose, tail_pose = image.split("_")
                image = self.queue.get()
                self.res.append(
                    {
                        "HEAD": head_pose.split(".")[1],
                        "LEG": leg_pose.split(".")[1],
                        "WING": wing_pose.split(".")[1],
                        "TAIL": tail_pose.split(".")[1],
                    }
                )
        except Exception as exc:  # noqa: BLE001 - captured to re-raise on the caller's thread
            self.error = exc
        finally:
            # Always release the lock, success or failure, so get_details()
            # never blocks on lock.acquire() forever.
            self.lock.release()

    def run(self):
        self.extract()

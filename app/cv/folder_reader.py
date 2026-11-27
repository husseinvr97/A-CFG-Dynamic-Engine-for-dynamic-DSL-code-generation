from threading import Thread
import os

class FolderReader(Thread):
    def __init__(self,path,queue, sentinel):
        super().__init__(daemon=True)
        self.path = path
        self.queue = queue
        self.sentinel = sentinel
        # Set if read() raises, so the caller (see operation_main.get_details)
        # can re-raise it on the main thread after the consumer unblocks.
        self.error: Exception | None = None

    def read(self):
        try:
            directory = self.path
            for image in os.listdir(directory):
                if image.endswith(".jpg"):
                    # img_num = image[:4]
                    img_name = image[4:-4]
                    self.queue.put(img_name)
        except Exception as exc:  # noqa: BLE001 - captured to re-raise on the caller's thread
            self.error = exc
        finally:
            # Always push the sentinel, success or failure, so the consumer
            # thread (DetailsExtractor) never blocks on queue.get() forever.
            self.queue.put(self.sentinel)

    def run(self):
        self.read()

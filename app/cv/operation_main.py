import threading
from queue import Queue

import pandas as pd
from app.cv.details_extractor import DetailsExtractor
from app.cv.folder_reader import FolderReader


def get_details(folderPath) -> list[dict[str, str]]:
    lock = threading.Semaphore(0)
    SENTINEL = object()  # SENTINEL OBJECT
    queue = Queue()
    res = list[dict[str, str]]()
    reader = FolderReader(folderPath, queue, SENTINEL)
    extractor = DetailsExtractor(queue, SENTINEL, lock, res)
    reader.start()
    extractor.start()
    lock.acquire()
    # Both threads are guaranteed to have finished by now: the extractor only
    # releases `lock` from its own finally block, after it has stopped
    # reading from `queue`. These joins just make that explicit and bound
    # (instead of relying on scheduling) before we inspect .error below.
    reader.join()
    extractor.join()
    if reader.error is not None:
        raise reader.error
    if extractor.error is not None:
        raise extractor.error
    return res


def filterColumns(colunms: list, data):
    wantHead = colunms.__contains__("head")
    wantLeg = colunms.__contains__("leg")
    wantWing = colunms.__contains__("wing")
    wantTail = colunms.__contains__("tail")
    res = []
    for item in data:
        tempDict = {}
        if wantHead:
            tempDict["HEAD"] = item["HEAD"]
        if wantLeg:
            tempDict["LEG"] = item["LEG"]
        if wantWing:
            tempDict["Wing"] = item["Wing"]
        if wantTail:
            tempDict["TAIL"] = item["TAIL"]
        res.append(tempDict)
    return pd.DataFrame(res)

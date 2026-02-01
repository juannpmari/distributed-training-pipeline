import threading


class Backpressure:
    def __init__(self, max_inflight: int):
        self.sem = threading.Semaphore(max_inflight)

    def acquire(self):
        self.sem.acquire()

    def release(self):
        self.sem.release()

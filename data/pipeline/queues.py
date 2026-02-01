import queue


class BoundedQueue:
    def __init__(self, maxsize: int):
        self.q = queue.Queue(maxsize=maxsize)

    def put(self, item):
        self.q.put(item)

    def get(self):
        return self.q.get()

import time

class Stopwatch:
    def __init__(self):
        self._elapsed = 0.0
        self._start_time = -1.0
        self._label = None

    def start(self, label: str | None = None):
        self._label = label
        self._elapsed = 0.0
        self._start_time = time.time()

    def pause(self):
        stop_time = time.time()
        if self._start_time > 0.0:
            self._elapsed += stop_time - self._start_time
            self._start_time = -1.0

    def resume(self):
        self._start_time = time.time()

    def stop(self):
        stop_time = time.time()
        if self._start_time > 0.0:
            self._elapsed += stop_time - self._start_time
            self._start_time = -1.0
        text = 'stopwatch: ' if self._label is None else f'stopwatch {self._label}: '
        text += f'{self._elapsed} s'
        print(text)
        return self._elapsed

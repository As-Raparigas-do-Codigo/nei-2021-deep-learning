import cv2
import copy
from threading import Thread, Lock
from typing import List
from imutils.video import VideoStream


class VideoCapture:

    def __init__(self, callback_functions: List, device_index=0, sleep_time=1):
        self._callback_functions = callback_functions
        self._device_index = device_index
        self._sleep_time = sleep_time
        self._mutex = Lock()
        self._stop_thread = False
        self._camera = None
        self._thread = None

    def _thread_video_frames(self):
        keep_running = True
        while keep_running:
            self._mutex.acquire()
            if self._stop_thread:
                keep_running = False
            self._mutex.release()

            frame = self._camera.read()
            frame = cv2.flip(frame, 1)
            for func in self._callback_functions:
                func(copy.deepcopy(frame))

            cv2.waitKey(self._sleep_time)

    def start(self):
        self._camera = VideoStream(usePiCamera=False, src=self._device_index).start()
        if self._camera is None:
            raise IndexError("Invalid device index.")

        if self._thread is not None:
            self.stop()

        self._stop_thread = False

        self._thread = Thread(target=self._thread_video_frames)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return

        self._mutex.acquire()
        self._stop_thread = True
        self._mutex.release()

        self._thread.join()
        self._thread = None
        self._camera.stop()
        self._camera = None

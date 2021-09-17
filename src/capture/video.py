import cv2
import copy
from threading import Thread, Lock
from typing import List


class VideoCapture:

    def __init__(self, callback_functions: List, device_index=0, sleep_time=1):
        self.callback_functions = callback_functions
        self.device_index = device_index
        self.sleep_time = sleep_time
        self.mutex = Lock()
        self.stop_thread = False
        self.camera = None
        self.thread = None

    def _thread_video_frames(self):
        keep_running = True
        while keep_running:
            self.mutex.acquire()
            if self.stop_thread:
                keep_running = False
            self.mutex.release()

            success, frame = self.camera.read()
            if success:
                frame = cv2.flip(frame, 1)
                for func in self.callback_functions:
                    func(copy.deepcopy(frame))
            else:
                keep_running = False

            cv2.waitKey(self.sleep_time)

    def start(self):
        self.camera = cv2.VideoCapture(self.device_index)
        if self.camera is None or not self.camera.isOpened():
            raise IndexError("Invalid device index.")

        if self.thread is not None:
            self.stop()

        self.stop_thread = False

        self.thread = Thread(target=self._thread_video_frames)
        self.thread.start()

    def stop(self):
        if self.thread is None:
            return

        self.mutex.acquire()
        self.stop_thread = True
        self.mutex.release()

        self.thread.join()
        self.thread = None
        self.camera.release()
        self.camera = None

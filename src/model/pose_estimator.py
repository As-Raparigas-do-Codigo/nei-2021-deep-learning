from .base import MAXModel
import cv2
import json
from threading import Lock, Thread
import copy


class PoseEstimator(MAXModel):

    def __init__(self, host="localhost", port=5000):
        super().__init__(host=host, port=port)
        self._mutex_frame = Lock()
        self._last_frame = None
        self._mutex_result = Lock()
        self._last_result = None
        self._thread = None

    def _thread_request(self, frame):
        response = self._post_image(frame, include_mime=True)
        result = json.loads(response.text)
        self._mutex_result.acquire()
        self._last_result = result
        self._mutex_result.release()

    def frame_callback(self, frame):
        color = (36, 255, 12)

        if self._thread is None or not self._thread.is_alive():
            self._thread = Thread(target=self._thread_request, args=(copy.deepcopy(frame),))
            self._thread.start()

        result = self._get_last_result()

        if result is not None and result["status"] == "ok":
            for pred in result["predictions"]:
                for pl in pred["pose_lines"]:
                    x1, y1, x2, y2 = pl["line"]
                    frame = cv2.line(frame, (x1, y1), (x2, y2), color, 2)

        _, frame = cv2.imencode('.jpg', frame)
        self._mutex_frame.acquire()
        self._last_frame = frame
        self._mutex_frame.release()

    def _get_last_result(self):
        self._mutex_result.acquire()
        cp_result = copy.deepcopy(self._last_result)
        self._mutex_result.release()
        return cp_result

    def get_last_frame(self):
        self._mutex_frame.acquire()
        cp_frame = copy.deepcopy(self._last_frame)
        self._mutex_frame.release()
        return cp_frame

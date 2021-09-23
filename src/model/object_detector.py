from .base import MAXModel
import cv2
import json
import os
from threading import Lock, Thread
from time import sleep
import copy


class ObjectDetector(MAXModel):

    _OUTPUT_JSON_FILE_NAME = "object_detector.json"
    _LIVE_MODE = 1
    _CLASSIFICATION_MODE = 2
    _DELAY_MODE = 3

    def __init__(self, host="localhost", port=5000):
        super().__init__(host=host, port=port)
        self._mutex_frame = Lock()
        self._last_frame_jpg = None
        self._last_frame_orig = None
        self._mutex_result = Lock()
        self._last_result = None
        self._thread = None
        self._mode = self._LIVE_MODE
        self._key = False
        self._mutex_key = Lock()

    def _thread_request(self, frame):
        response = self._post_image(frame, {"threshold": 0.7})
        result = json.loads(response.text)
        self._mutex_result.acquire()
        self._last_result = result
        self._mutex_result.release()

    def frame_callback(self, frame):
        color = (36, 255, 12)
        height, width = frame.shape[:2]

        if self._mode == self._DELAY_MODE:
            sleep(5)
            file = open(os.path.join(self._OUTPUT_DIR, self._OUTPUT_JSON_FILE_NAME), "w")
            file.write(json.dumps({"labels": []}))
            file.flush()
            file.close()
            self._mode = self._LIVE_MODE

        if self._get_key() and self._mode == self._LIVE_MODE:
            self._mode = self._CLASSIFICATION_MODE
            self._mutex_key.acquire()
            self._key = False
            self._mutex_key.release()
            self._thread = Thread(target=self._thread_request, args=(copy.deepcopy(frame),))
            self._thread.start()

        if self._mode == self._CLASSIFICATION_MODE:
            self._mutex_frame.acquire()
            frame = self._last_frame_orig
            self._mutex_frame.release()

            result = self._get_last_result()

            if result is not None and result["status"] == "ok":
                ignore_repeated_predictions = []
                for pred in result["predictions"]:
                    if pred['label'] not in ignore_repeated_predictions:
                        y_min, x_min, y_max, x_max = pred["detection_box"]
                        x = int(x_min * width)
                        y = int(y_min * height)
                        x2 = int(x_max * width)
                        y2 = int(y_max * height)
                        frame = cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
                        ignore_repeated_predictions.append(pred['label'])

                ignore_repeated_predictions = []
                for pred in result["predictions"]:
                    if pred['label'] not in ignore_repeated_predictions:
                        y_min, x_min, y_max, x_max = pred["detection_box"]
                        x = int(x_min * width)
                        y = int(y_min * height)
                        text_position = (x + 2, y + 14)
                        label = f"{pred['label'].capitalize()}: {round(pred['probability'] * 100)}%"
                        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                        frame = cv2.rectangle(frame, (x, y), (x + text_w + 5, y + text_h + 12), color, -1)
                        cv2.putText(frame, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                        ignore_repeated_predictions.append(pred['label'].replace(" ", "_"))

                file = open(os.path.join(self._OUTPUT_DIR, self._OUTPUT_JSON_FILE_NAME), "w")
                file.write(json.dumps({"labels": ignore_repeated_predictions}))
                file.flush()
                file.close()

                self._last_result = None
                self._mode = self._DELAY_MODE

        self._mutex_frame.acquire()
        self._last_frame_orig = copy.deepcopy(frame)
        _, frame = cv2.imencode('.jpg', frame)
        self._last_frame_jpg = frame
        self._mutex_frame.release()

    def space_bar(self):
        self._mutex_key.acquire()
        self._key = True
        self._mutex_key.release()

    def _get_key(self):
        self._mutex_key.acquire()
        cp_key = self._key
        self._mutex_key.release()
        return cp_key

    def _get_last_result(self):
        self._mutex_result.acquire()
        cp_result = copy.deepcopy(self._last_result)
        self._mutex_result.release()
        return cp_result

    def get_last_frame(self):
        self._mutex_frame.acquire()
        cp_frame = copy.deepcopy(self._last_frame_jpg)
        self._mutex_frame.release()
        return cp_frame

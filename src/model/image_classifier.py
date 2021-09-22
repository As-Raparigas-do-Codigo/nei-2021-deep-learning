from .base import MAXModel
import cv2
import json


class ResNetImageClassifier(MAXModel):

    def __init__(self, host="localhost", port=5000, resize_dim=None):
        super().__init__(host=host, port=port)
        self._resize_dim = resize_dim

    def frame_callback(self, frame):
        color = (36, 255, 12)
        position = (15, 35)
        orig_frame = frame

        if self._resize_dim is not None:
            frame = cv2.resize(frame, self._resize_dim)

        response = self._post_image(frame)
        result = json.loads(response.text)
        if result["status"] == "ok" and len(result["predictions"]) > 0:
            if result["predictions"][0]["probability"] > 0.7:
                cv2.putText(frame, result["predictions"][0]["label"],
                            position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("ResNet Image Classifier", orig_frame)

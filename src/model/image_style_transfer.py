from .base import MAXModel
import numpy as np
import cv2


class FastNeuralStyleTransfer(MAXModel):

    def frame_callback(self, frame):
        response = self._post_image(frame, {"model": "mosaic"})
        if response.status_code == 200:
            decoded_image = cv2.imdecode(np.frombuffer(response.content, np.uint8), -1)
            cv2.imshow("Fast Neural Style Transfer", decoded_image)

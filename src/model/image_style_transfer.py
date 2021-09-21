from .base import MAXModel
import numpy as np
import cv2
import os


class FastNeuralStyleTransfer(MAXModel):

    _OUTPUT_FILE_NAME = "fast_neural_style_transfer.jpg"

    def frame_callback(self, frame):
        response = self._post_image(frame, {"model": "mosaic"})
        if response.status_code == 200:
            decoded_image = cv2.imdecode(np.frombuffer(response.content, np.uint8), -1)
            # cv2.imshow("Fast Neural Style Transfer", decoded_image)
            cv2.imwrite(os.path.join(self._OUTPUT_DIR, self._OUTPUT_FILE_NAME), decoded_image)

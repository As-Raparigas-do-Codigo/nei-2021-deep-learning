from .base import MAXModel
import cv2
import json
import os


class PoseEstimator(MAXModel):

    _OUTPUT_FILE_NAME = "pose_estimator.jpg"

    def frame_callback(self, frame):
        color = (36, 255, 12)

        response = self._post_image(frame, include_mime=True)
        result = json.loads(response.text)
        if result["status"] == "ok":
            for pred in result["predictions"]:
                for pl in pred["pose_lines"]:
                    x1, y1, x2, y2 = pl["line"]
                    frame = cv2.line(frame, (x1, y1), (x2, y2), color, 2)

        # cv2.imshow("Pose Estimator", frame)
        cv2.imwrite(os.path.join(self._OUTPUT_DIR, self._OUTPUT_FILE_NAME), frame)

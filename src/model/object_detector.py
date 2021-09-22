from .base import MAXModel
import cv2
import json
import os


class ObjectDetector(MAXModel):

    _OUTPUT_IMAGE_FILE_NAME = "object_detector.jpg"
    _OUTPUT_JSON_FILE_NAME = "object_detector.json"

    def frame_callback(self, frame):
        color = (36, 255, 12)
        height, width = frame.shape[:2]

        response = self._post_image(frame, {"threshold": 0.7})
        result = json.loads(response.text)
        if result["status"] == "ok":
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

        # cv2.imshow("Object Detector", frame)
        cv2.imwrite(os.path.join(self._OUTPUT_DIR, self._OUTPUT_IMAGE_FILE_NAME), frame)

from .base import MAXModel
import json


class AudioClassifier(MAXModel):

    def frame_callback(self, frame):
        response = self._post_audio(frame, {"start_time": 0})
        print(json.loads(response.text))

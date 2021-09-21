from .base import MAXModel
import os


class AudioClassifier(MAXModel):

    _OUTPUT_FILE_NAME = "audio_classifier.json"

    def frame_callback(self, frame):
        response = self._post_audio(frame, {"start_time": 0})
        file = open(os.path.join(self._OUTPUT_DIR, self._OUTPUT_FILE_NAME), "w")
        file.write(response.text)
        file.flush()
        file.close()

from abc import ABC, abstractmethod
from tempfile import NamedTemporaryFile
import cv2
import requests
import os


class BaseModel(ABC):

    _OUTPUT_DIR = os.path.join("web", "output")

    def __init__(self):
        if not os.path.exists(self._OUTPUT_DIR):
            os.makedirs(self._OUTPUT_DIR)

    @abstractmethod
    def frame_callback(self, frame):
        pass


class MAXModel(BaseModel, ABC):

    def __init__(self, host="localhost", port=5000):
        super().__init__()
        self._host = host
        self._port = port
        self._url = f'http://{self._host}:{self._port}/model/predict'

    def _post_image(self, frame, query_params=None, include_mime=False):
        temp_file = NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(temp_file.name, frame)
        temp_file.close()
        f = open(temp_file.name, "rb")
        if include_mime:
            file_form = {'file': (temp_file.name, f, 'image/jpeg')}
        else:
            file_form = {'image': f}
        if query_params is None:
            query_params = {}
        response = requests.post(self._url, params=query_params, files=file_form)
        f.close()
        os.unlink(temp_file.name)
        return response

    def _post_audio(self, audio_path, query_params=None):
        f = open(audio_path, "rb")
        if query_params is None:
            query_params = {}
        files = {'audio': (audio_path, f, 'audio/wav')}
        response = requests.post(self._url, params=query_params, files=files)
        f.close()
        os.unlink(audio_path)
        return response

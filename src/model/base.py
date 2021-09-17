from abc import ABC, abstractmethod
from tempfile import NamedTemporaryFile
import cv2
import requests
import os


class BaseModel(ABC):

    @abstractmethod
    def frame_callback(self, frame):
        pass


class MAXModel(BaseModel, ABC):

    def __init__(self, host="localhost", port=5000):
        self.host = host
        self.port = port
        self.url = f'http://{self.host}:{self.port}/model/predict'

    def _post_image(self, frame, query_params=None):
        temp_file = NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(temp_file.name, frame)
        temp_file.close()
        f = open(temp_file.name, "rb")
        files = {'image': f}
        if query_params is None:
            query_params = {}
        response = requests.post(self.url, params=query_params, files=files)
        f.close()
        os.unlink(temp_file.name)
        return response

    def _post_audio(self, audio_path, query_params=None):
        f = open(audio_path, "rb")
        if query_params is None:
            query_params = {}
        files = {'audio': (audio_path, f, 'audio/wav')}
        response = requests.post(self.url, params=query_params, files=files)
        f.close()
        os.unlink(audio_path)
        return response

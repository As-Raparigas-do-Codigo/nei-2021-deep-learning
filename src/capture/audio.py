from threading import Thread, Lock
from tempfile import NamedTemporaryFile
from typing import List
import pyaudio
import wave


class AudioCapture:
    _FORMAT = pyaudio.paInt16
    _CHANNELS = 2
    _RATE = 44100
    _CHUNK = 1024
    _RECORD_SECONDS = 3

    def __init__(self, callback_functions: List):
        self.callback_functions = callback_functions
        self.mutex = Lock()
        self.stop_thread = False
        self.thread = None

    def _thread_audio_record(self):
        keep_running = True
        while keep_running:
            self.mutex.acquire()
            if self.stop_thread:
                keep_running = False
            self.mutex.release()

            audio = pyaudio.PyAudio()
            stream = audio.open(format=self._FORMAT, channels=self._CHANNELS,
                                rate=self._RATE, input=True,
                                frames_per_buffer=self._CHUNK)

            frames = []
            for i in range(0, int(self._RATE / self._CHUNK * self._RECORD_SECONDS)):
                data = stream.read(self._CHUNK)
                frames.append(data)

            stream.stop_stream()
            stream.close()
            audio.terminate()

            temp_file = NamedTemporaryFile(suffix=".wav", delete=False)
            wave_file = wave.open(temp_file, 'wb')
            wave_file.setnchannels(self._CHANNELS)
            wave_file.setsampwidth(audio.get_sample_size(self._FORMAT))
            wave_file.setframerate(self._RATE)
            wave_file.writeframes(b''.join(frames))
            wave_file.close()
            temp_file.close()

            for func in self.callback_functions:
                func(temp_file.name)

    def start(self):
        if self.thread is not None:
            self.stop()

        self.stop_thread = False

        self.thread = Thread(target=self._thread_audio_record)
        self.thread.start()

    def stop(self):
        if self.thread is None:
            return

        self.mutex.acquire()
        self.stop_thread = True
        self.mutex.release()

        self.thread.join()
        self.thread = None

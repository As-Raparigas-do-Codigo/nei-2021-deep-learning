from capture import *
from model import *

if __name__ == '__main__':
    # video_clf_model = ResNetImageClassifier()
    # video_gen_model = FastNeuralStyleTransfer(port=5002)
    video_clf_model = ObjectDetector(port=5001)
    video_cap = VideoCapture([video_clf_model.frame_callback], sleep_time=50)
    video_cap.start()

    audio_clf_model = AudioClassifier(port=5003)
    audio_cap = AudioCapture([audio_clf_model.frame_callback])
    audio_cap.start()

    while True:
        key = input("Press 'q' to quit...")
        if key == "q":
            break

    video_cap.stop()
    audio_cap.stop()

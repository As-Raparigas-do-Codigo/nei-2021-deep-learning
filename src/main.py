from capture import *
from model import *
import docker
from time import sleep


if __name__ == '__main__':
    docker_client = docker.from_env()
    print("Starting Docker containers...")
    containers = [
        # docker run -it -p 5001:5000 max-object-detector
        docker_client.containers.run("max-object-detector", detach=True, ports={5000: 5001}),
        # docker run -it -p 5002:5000 max-human-pose-estimator
        docker_client.containers.run("max-human-pose-estimator", detach=True, ports={5000: 5002}),
        # docker run -it -p 5003:5000 max-audio-classifier
        docker_client.containers.run("max-audio-classifier", detach=True, ports={5000: 5003})
    ]

    for c in containers:
        updated_c = c
        elapsed_time = 0
        while updated_c.status != 'running':
            sleep(1)
            elapsed_time += 1
            if elapsed_time == 30:
                raise RuntimeError("Unable to start all Docker containers.")
            updated_c = docker_client.containers.get(c.id)

    obj_detector_model = ObjectDetector(port=5001)
    pose_estimator_model = PoseEstimator(port=5002)
    print("Starting video capture...")
    video_cap = VideoCapture([obj_detector_model.frame_callback, pose_estimator_model.frame_callback], sleep_time=90)
    video_cap.start()

    audio_clf_model = AudioClassifier(port=5003)
    print("Starting audio capture...")
    audio_cap = AudioCapture([audio_clf_model.frame_callback])
    audio_cap.start()

    print("All systems are up and running...")
    while True:
        key = input("Press 'q' to quit...")
        if key == "q":
            break

    print("Stopping video capture...")
    video_cap.stop()
    print("Stopping audio capture...")
    audio_cap.stop()

    print("Stopping Docker containers...")
    for c in containers:
        c.stop()

from capture import *
from model import *
import docker
from flask import Flask, render_template, send_file, json, Response
from time import sleep

_SLEEP_TIME = 200

app = Flask(__name__, template_folder="./web")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/logo')
def get_logo_image():
    return send_file("./web/logo.png", mimetype='image/png')

@app.route('/cocoicons/<file>')
def get_coco_icon(file):
    return send_file(f"./web/cocoicons/{file}", mimetype='image/jpg')

@app.route('/object_detector_labels')
def get_object_detector_labels():
    with open("./web/output/object_detector.json") as json_file:
        return json.load(json_file)

@app.route('/audio_classifier')
def get_audio_classifier_labels():
    with open("./web/output/audio_classifier.json") as json_file:
        return json.load(json_file)

def gen_frames(model):
    while True:
        frame = model.get_last_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')
        sleep(_SLEEP_TIME / 1000)

@app.route('/video_object_detector')
def video_object_detector():
    return Response(gen_frames(obj_detector_model), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_pose_estimator')
def video_pose_estimator():
    return Response(gen_frames(pose_estimator_model), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/space_bar', methods=['POST'])
def post_space_bar():
    obj_detector_model.space_bar()
    return Response("", 204)


if __name__ == '__main__':
    docker_client = docker.from_env()
    print("Starting Docker containers...")
    containers = [
        # docker run -it -p 5001:5000 max-object-detector
        # docker_client.containers.run("max-object-detector", detach=True, ports={5000: 5001}),
        docker_client.containers.run("quay.io/codait/max-object-detector:intel-x86_64-v2.0.0-faster_rcnn_resnet101", detach=True, ports={5000: 5001}),
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

    # 'max-audio-classifier' needs a 5 second delay...
    sleep(5)

    obj_detector_model = ObjectDetector(port=5001)
    pose_estimator_model = PoseEstimator(port=5002)
    print("Starting video capture...")
    video_cap = VideoCapture([obj_detector_model.frame_callback, pose_estimator_model.frame_callback],
                             sleep_time=_SLEEP_TIME, device_index=0)
    video_cap.start()

    audio_clf_model = AudioClassifier(port=5003)
    print("Starting audio capture...")
    audio_cap = AudioCapture([audio_clf_model.frame_callback])
    audio_cap.start()

    print("All systems are up and running...")
    app.run(debug=False, host='0.0.0.0', port=8081)

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

from flask import Flask, render_template, send_file, json

app = Flask(__name__, template_folder=".")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/logo')
def get_logo_image():
    return send_file("./logo.png", mimetype='image/png')


@app.route('/cocoicons/<file>')
def get_coco_icon(file):
    return send_file(f"./cocoicons/{file}", mimetype='image/jpg')


@app.route('/object_detector')
def get_object_detector_image():
    return send_file("./output/object_detector.jpg", mimetype='image/jpg')


@app.route('/object_detector_labels')
def get_object_detector_labels():
    with open("./output/object_detector.json") as json_file:
        return json.load(json_file)


@app.route('/pose_estimator')
def get_pose_estimator_image():
    return send_file("./output/pose_estimator.jpg", mimetype='image/jpg')


@app.route('/fast_neural_style_transfer')
def get_fast_neural_style_transfer_image():
    return send_file("./output/fast_neural_style_transfer.jpg", mimetype='image/jpg')


@app.route('/audio_classifier')
def get_audio_classifier_json():
    with open("./output/audio_classifier.json") as json_file:
        return json.load(json_file)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)

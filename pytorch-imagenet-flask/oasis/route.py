import io
import json
from oasis import app
from flask import send_file, render_template
from flask import Flask, jsonify, request, render_template

from oasis.model.imagenet import Densenet121

model = Densenet121()


@app.route("/", endpoint="index")
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = model.get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})


@app.route("/finger", methods=["GET"])
def finger_get():
    return send_file("static/img/finger.jpg", mimetype='image/jpg')

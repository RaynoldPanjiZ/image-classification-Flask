from email import message
from flask import Flask, jsonify, request, redirect, url_for, render_template, flash
import keras
from keras_preprocessing import image
# from keras.models import load_model
import os
import numpy as np
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'


#### Front End ####

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def upload():
    file = request.files['file']
    if check_format(file) == 0:
        msg = "Allowed image types are -> png, jpg, jpeg, gif"
        return render_template('index.html', message=msg)
        
    # filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    path = os.path.join(request.url_root, file_path)
    pred = classify(file_path)
    return render_template('index.html', filename=file.filename, pred=pred, path=path)



#### cek format gambar ####

def check_format(file):
    formats = set(['png', 'jpg', 'jpeg', 'gif'])
    extention = '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in formats
    if file and extention:
        return 1
    else:
	    return 0


#### tampilkan gambar ####

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/'+filename), code=301)



#### API response (Backend) ####

@app.route('/api/predict', methods=["POST"])
def predict():

    if request.method == "POST":
        file = request.files['file']
        check_format(file)
        if check_format(file) == 0:
            res = {
                'filename': file.filename,
                'message': "Allowed image types are -> png, jpg, jpeg, gif"
            }
        else:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            pred = classify(file_path)

            res = {
                'filename': file.filename,
                'path': os.path.join(request.host_url, file_path),
                'prediksi': pred[1],
                'kemiripan': str(pred[0]), 
                'semua_kemiripan': pred[2]
            }

        return jsonify(res)
    return jsonify(request='404')
    


#### Model Prediction ####

def convert_image(path):
    img = image.load_img(path, target_size=(150, 150))

    x = image.img_to_array(img)
    x = np.array(x)/255.0
    x = np.expand_dims(x, axis=0)
    return np.vstack([x])


def classify(path):
    images = convert_image(path)

    labels = json.load(open('model/labels_map.json'))
    class_names = list(labels.keys())

    savedModel = keras.models.load_model('model/model.h5')
    pred = savedModel.predict(images)

    proba = np.max(pred[0], axis=-1)
    proba_all = {b:str(a) for a,b in zip(pred[0], class_names)}
    predicted_class = class_names[np.argmax(pred[0], axis=-1)]
    # if proba >= 0.5:
    #     predicted_class = class_names[np.argmax(pred[0], axis=-1)]
    # else:
    #     predicted_class = "tidak ada yang mirip"
    #     proba = 0

    prediction = [proba, predicted_class, proba_all]
    return prediction


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
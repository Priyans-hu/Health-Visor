import glob
import joblib
import numpy as np
import os
import pickle
from PIL import Image
import re
import sys
import tensorflow as tf 

from flask import Flask, redirect, url_for, request, render_template,flash,redirect
from flask import request
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from werkzeug.utils import secure_filename


app = Flask(__name__)
MODEL_PATH = 'models/BrainTumour.h5'
model = load_model(MODEL_PATH)

MODEL_PATH2 = 'models/Pneumonia.h5'
model2 = load_model(MODEL_PATH2)
print('Model loaded. Start serving...')

def predict_label(img_path, model):
    img = image.load_img(img_path, target_size=(200,200)) 
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')/255
    preds = model.predict(img)
    pred = np.argmax(preds,axis = 1)
    str0 = ''
    if pred[0] == 0:
        str0 = "Glioma"
    elif pred[0] == 1:
        str0 = 'Meningioma'
    elif pred[0]==3:
        str0 = 'Pituitary'
    else:
        str0 = "Normal"
    return str0

def model_predict(img_path):
	img = tf.keras.utils.load_img(img_path, target_size=(64, 64))
	img = tf.keras.preprocessing.image.img_to_array(img)
	img = np.expand_dims(img, axis=0)
	preds = model2.predict(img)
	if preds==1:
		preds ="Pneumonia"
	else:
		preds="Normal"
	return preds


@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("main.html")

@app.route("/index", methods=['GET', 'POST'])
def index():
	return render_template("index.html")

@app.route("/index2", methods=['GET', 'POST'])
def index2():
	return render_template("index2.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']
		img_path = "static/brain_uploads" + img.filename	
		img.save(img_path)
		p = predict_label(img_path,model)
	return render_template("index.html", prediction = p, img_path = img_path)


@app.route("/predict", methods = ['GET', 'POST'])
def output():
	if request.method == 'POST':
		img = request.files['my_image']
		img_path = "static/pneumonia_uploads" + img.filename	
		img.save(img_path)
		p = model_predict(img_path)
	return render_template("index2.html", prediction = p, img_path = img_path)

if __name__ == '__main__':
        app.run(host="0.0.0.0",port=8000)
    

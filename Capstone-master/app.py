import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

def load_image(img_path_):
    img_ = image.load_img(img_path_, target_size=(50, 50))
    img_tensor = image.img_to_array(img_)  # (height, width, channels)
    # img_tensor = img_tensor.reshape(1, 50, 50, 3)
    img_tensor = np.expand_dims(img_tensor,
                                axis=0)  # (1, height, width, channels), add a dimension because the model expects
    # this shape: (batch_size, height, width, channels)
    img_tensor /= 255.  # imshow expects values in the range [0, 1]

    return img_tensor


def prediction(img_path_):
    new_image = load_image(img_path_)

    # Load the model
    model = load_model('mymodel.h5')
    print("Model is loaded")

    pred = model.predict(new_image)

    labels = np.array(pred)
    predicted_value = labels[0][1]

    THRESHOLD_VALUE = float(8.09e-11)

    if predicted_value > THRESHOLD_VALUE:
        return "Class 0 - No Cancer!"
    else:
        return "Class 1 - Possibility of having Cancer!"


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("home.html")


@app.route("/about")
def about_page():
    return "This is the Breast Cancer Detection Web Application!"


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    global img_path, p
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/" + img.filename
        img.save(img_path)

        p = prediction(img_path)

    return render_template("home.html", prediction=p, img_path=img_path)


if __name__ == "__main__":
    # app.debug = True
    app.run(debug=True)

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

global model


def load_image(img_path):
    img_ = image.load_img(img_path, target_size=(50, 50))
    img_tensor = image.img_to_array(img_)  # (height, width, channels)
    # img_tensor = img_tensor.reshape(1, 50, 50, 3)
    img_tensor = np.expand_dims(img_tensor,
                                axis=0)  # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.  # imshow expects values in the range [0, 1]

    return img_tensor


def prediction(img_path):
    new_image = load_image(img_path)

    pred = model.predict(new_image)

    labels = np.array(pred)
    predicted_value = labels[0][1]
    print(predicted_value)
    THRESHOLD_VALUE = float(8.09e-11)

    if predicted_value > THRESHOLD_VALUE:
        return "Class 0"
    else:
        return "Class 1"


if __name__ == "__main__":
    # Load the model
    model = load_model('mymodel.h5')
    print("Model is loaded")

    # Model prediction
    image_path = r"C:\Users\sanma\PycharmProjects\BCDetection\Test_images\0\10276_idx5_x251_y1201_class0.png"
    pres_class = prediction(image_path)

    print(pres_class)





import numpy as np
from flask import Flask, request, render_template
import pickle

Flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@Flask_app.route("/")
def home():
    return render_template("index.html", prediction_text="")  # FIXED

@Flask_app.route("/predict", methods=["POST"])
def product():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text="The Predicted crop is {}".format(prediction[0]))

if __name__ == "__main__":
    Flask_app.run(debug=True)

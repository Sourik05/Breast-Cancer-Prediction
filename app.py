import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("Breast_Cancer_Detector.pkl", "rb"))


@flask_app.route("/")
def home():
    return render_template("home.html")


@flask_app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    df = pd.DataFrame(features)
    output = model.predict(df)

    if output == 1:
        res_val = "breast cancer"
    else:
        res_val = "no breast cancer"

    print(res_val)
    return render_template("output.html", prediction_text=res_val)


if __name__ == "__main__":
    flask_app.run(debug=True)

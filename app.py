from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load pipeline

pipeline = joblib.load("loan_approval_pipeline.pkl")

model = pipeline["model"]
features = pipeline["features"]
imputer = pipeline["imputer"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():


    try:

        data = request.json

        df = pd.DataFrame([data])

        df = df[features]

        df = pd.DataFrame(
            imputer.transform(df),
            columns=features
        )

        prediction = int(model.predict(df)[0])

        default_probability = float(
            model.predict_proba(df)[0][1]
        )

        return jsonify({
            "prediction": prediction,
            "default_probability": default_probability,
            "success": True
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        })


if __name__ == "__main__":
    app.run(debug=True)

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

        # =========================
        # SAFE VALUES
        # =========================

        income = max(float(df["person_income"].iloc[0]), 1)
        emp = max(float(df["person_emp_length"].iloc[0]), 1)
        cred = max(float(df["cb_person_cred_hist_length"].iloc[0]), 1)
        age = max(float(df["person_age"].iloc[0]), 1)

        df["loan_percent_income"] = (
            df["loan_amnt"] / income)

        # =========================
        # FEATURE ENGINEERING
        # =========================

        df["loan_to_income_ratio"] = df["loan_amnt"] / income

        df["financial_burden"] = (
            df["loan_amnt"] * df["loan_int_rate"]
        )

        df["income_per_year_emp"] = (
            income / emp
        )

        df["cred_hist_to_age_ratio"] = (
            cred / age
        )

        df["int_to_loan_ratio"] = (
            df["loan_int_rate"] / df["loan_amnt"]
        )

        df["loan_int_emp_interaction"] = (
            df["loan_int_rate"] * emp
        )

        df["debt_to_credit_ratio"] = (
            df["loan_amnt"] / cred
        )

        df["int_to_cred_hist"] = (
            df["loan_int_rate"] / cred
        )

        df["int_per_year_emp"] = (
            df["loan_int_rate"] / emp
        )

        df["loan_amt_per_emp_year"] = (
            df["loan_amnt"] / emp
        )

        df["income_to_loan_ratio"] = (
            income / df["loan_amnt"]
        )

        # =========================
        # MATCH TRAINING FEATURES
        # =========================

        df = df[features]

        df = pd.DataFrame(
            imputer.transform(df),
            columns=features
        )

        prediction = int(
            model.predict(df)[0]
        )

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
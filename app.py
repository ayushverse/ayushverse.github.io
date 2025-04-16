from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load model and encoder
model = joblib.load("f1_lr.pkl")
category_encoder = joblib.load("c1.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect form input
        amt = float(request.form["amt"])
        category = request.form["category"]
        datetime_str = request.form["datetime"]
        city_pop = int(request.form["city_pop"])
        merch_lat = float(request.form["merch_lat"])
        merch_long = float(request.form["merch_long"])

        # Convert datetime string to Unix timestamp
        try:
            dt = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M")  # browser native format
        except ValueError:
            dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")  # flatpickr format

        unix_time = int(dt.timestamp())
        hour = dt.hour

        # Feature engineering
        high_amt_flag = int(amt > 500)
        low_city_pop_flag = int(city_pop < 5000)
        amt_log = np.log1p(amt)
        amt_to_pop_ratio = amt / (city_pop + 1)
        unusual_hour_flag = int(hour in [0, 1, 2, 3, 4])
        risk_category_flag = int(category in ['travel', 'shopping_net', 'misc_net', 'health_fitness'])

        # Encode category safely
        if category not in category_encoder.classes_:
            return render_template("index.html", prediction=f"❌ Error: Unknown category '{category}'")
        encoded_category = category_encoder.transform([category])[0]

        # Prepare DataFrame for prediction
        input_df = pd.DataFrame([[amt_log, encoded_category, unix_time, city_pop,
                                  merch_lat, merch_long, hour, high_amt_flag,
                                  low_city_pop_flag, amt_to_pop_ratio,
                                  unusual_hour_flag, risk_category_flag]],
                                columns=[
                                    'amt_log', 'category', 'unix_time', 'city_pop',
                                    'merch_lat', 'merch_long', 'hour',
                                    'high_amt_flag', 'low_city_pop_flag',
                                    'amt_to_pop_ratio', 'unusual_hour_flag',
                                    'risk_category_flag'
                                ])

        # Predict probability
        prob = model.predict_proba(input_df)[0][1]

        # Interpret fraud risk level
        if prob > 0.50:
            label = "🔥 LIKELY FRAUD"
        elif prob > 0.20:
            label = "⚠️ POSSIBLE FRAUD"
        else:
            label = "✅ LOW RISK"

        result = f"{label} (Fraud Risk Score: {prob:.2%})"
        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction=f"❌ Error: {str(e)}")

@app.route("/team")
def team():
    return render_template("team.html")

if __name__ == "__main__":
    app.run(debug=True)

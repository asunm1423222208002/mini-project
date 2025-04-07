import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template

# Load models and scaler from pickle files
with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("gb_model.pkl", "rb") as f:
    gb_model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = ["Latitude", "Longitude", "Depth", "Nst", "Gap", "Clo", "RMS"]
        user_input = pd.DataFrame([{feature: data[feature] for feature in features}])
        user_input_scaled = scaler.transform(user_input)
        
        rf_prediction = rf_model.predict(user_input_scaled)[0]
        gb_prediction = gb_model.predict(user_input_scaled)[0]
        
        return jsonify({
            "Random Forest Prediction": rf_prediction,
            "Gradient Boosting Prediction": gb_prediction
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)

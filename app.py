from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load model & dataset safely
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
data = pd.read_csv(os.path.join(BASE_DIR, "cleaned_dataset.csv"))

@app.route('/')
def home():
    if "session_id" in data.columns:
        session_ids = data["session_id"].astype(str).tolist()
    else:
        session_ids = data.index.astype(str).tolist()

    return render_template("index.html", session_ids=session_ids)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        session_id = request.form['session_id']

        # Select row
        if "session_id" in data.columns:
            row = data[data["session_id"].astype(str) == session_id]
        else:
            row = data.iloc[[int(session_id)]]

        if row.empty:
            if "session_id" in data.columns:
                session_ids = data["session_id"].astype(str).tolist()
            else:
                session_ids = data.index.astype(str).tolist()

            return render_template("index.html",
                                   prediction_text="Session not found",
                                   session_ids=session_ids)

        # Drop unnecessary columns
        drop_cols = ["session_id", "attack_detected"]
        row = row.drop(columns=[col for col in drop_cols if col in row.columns])

        final_input = row.values

        prediction = model.predict(final_input)

        result = "🚨 Intrusion Detected - Blocking IP" if prediction[0] == 1 else "✅ Normal Traffic"

        # Fix session_ids again here
        if "session_id" in data.columns:
            session_ids = data["session_id"].astype(str).tolist()
        else:
            session_ids = data.index.astype(str).tolist()

        return render_template("index.html",
                               prediction_text=result,
                               session_ids=session_ids)

    except Exception as e:
        if "session_id" in data.columns:
            session_ids = data["session_id"].astype(str).tolist()
        else:
            session_ids = data.index.astype(str).tolist()

        return render_template("index.html",
                               prediction_text=f"Error: {str(e)}",
                               session_ids=session_ids)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
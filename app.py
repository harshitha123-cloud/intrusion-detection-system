from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# ✅ Load model
model = joblib.load("model.pkl")

# ✅ Load CLEANED dataset (IMPORTANT)
data = pd.read_csv("cleaned_dataset.csv")

@app.route('/')
def home():
    # If session_id exists use it, else use index
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
            return render_template("index.html",
                                   prediction_text="Session not found",
                                   session_ids=data.index.astype(str).tolist())

        # Drop unnecessary columns
        drop_cols = ["session_id", "attack_detected"]
        row = row.drop(columns=[col for col in drop_cols if col in row.columns])

        # Final input
        final_input = row.values

        # Prediction
        prediction = model.predict(final_input)

        result = "🚨 Intrusion Detected - Blocking IP" if prediction[0] == 1 else "✅ Normal Traffic"

        return render_template("index.html",
                               prediction_text=result,
                               session_ids=data.index.astype(str).tolist())

    except Exception as e:
        return render_template("index.html",
                               prediction_text=f"Error: {str(e)}",
                               session_ids=data.index.astype(str).tolist())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
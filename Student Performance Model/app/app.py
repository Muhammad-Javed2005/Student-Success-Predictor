from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model & preprocessors
loaded_model = joblib.load("models/grade_model.pkl")
loaded_scaler = joblib.load("models/scaler.pkl")
loaded_le = joblib.load("models/label_encoder.pkl")
--
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    student_id = request.form['student_id']
    attendance = float(request.form['attendance'])
    midterm = float(request.form['midterm'])
    final = float(request.form['final'])
    study_hours = float(request.form['study_hours'])
    sleep_hours = float(request.form['sleep_hours'])

    # Features ka order model training ke hisaab se
    features = [[attendance, midterm, final, study_hours, sleep_hours]]
    features_scaled = loaded_scaler.transform(features)
    prediction = loaded_model.predict(features_scaled)
    grade = loaded_le.inverse_transform(prediction)[0]

    return render_template("result.html", name=name, student_id=student_id, grade=grade)

if __name__ == '__main__':
    app.run(debug=True)

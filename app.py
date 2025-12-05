
from flask import Flask, render_template, request, send_file
import joblib
import numpy as np
from fpdf import FPDF
import os
from datetime import datetime

app = Flask(__name__)

# Load trained model
model_path = os.path.join('model', 'disease_model.pkl')
try:
    model = joblib.load(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return render_template('error.html', error="Model not loaded properly")

        # Get form data
        age = float(request.form['age'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])

        # Validate inputs
        if not (0 <= age <= 150):
            return render_template('error.html', error="Age must be between 0 and 150")
        if not (0 <= glucose <= 500):
            return render_template('error.html', error="Glucose must be between 0 and 500")
        if not (0 <= blood_pressure <= 300):
            return render_template('error.html', error="Blood Pressure must be between 0 and 300")

        # Prepare features
        features = np.array([[age, glucose, blood_pressure]])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        # Convert to readable output
        result_text = "High Risk" if prediction == 1 else "Low Risk"
        risk_probability = probability[1] if prediction == 1 else probability[0]

        # Generate PDF report
        pdf_file = generate_pdf(age, glucose, blood_pressure, result_text, f"{risk_probability:.2%}")

        return render_template('result.html',
                               result=result_text,
                               report_file=pdf_file,
                               probability=f"{risk_probability:.2%}")

    except Exception as e:
        return render_template('error.html', error=f"Prediction error: {str(e)}")


def generate_pdf(age, glucose, blood_pressure, prediction, probability):
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Disease Prediction Report", ln=True, align='C')
    pdf.ln(10)

    # Timestamp
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(10)

    # Patient info
    pdf.cell(0, 10, f"Age: {age}", ln=True)
    pdf.cell(0, 10, f"Glucose: {glucose} mg/dL", ln=True)
    pdf.cell(0, 10, f"Blood Pressure: {blood_pressure} mmHg", ln=True)
    pdf.ln(10)

    # Prediction result
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Prediction: {prediction}", ln=True)
    pdf.cell(0, 10, f"Confidence: {probability}", ln=True)

    # Risk interpretation
    pdf.ln(10)
    pdf.set_font("Arial", "I", 10)
    if prediction == "High Risk":
        pdf.cell(0, 10, "Recommendation: Please consult with a healthcare professional.", ln=True)
    else:
        pdf.cell(0, 10, "Recommendation: Maintain regular health checkups.", ln=True)

    # Ensure reports folder exists
    if not os.path.exists("reports"):
        os.makedirs("reports")

    # Save PDF
    file_name = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    full_path = os.path.join("reports", file_name)
    pdf.output(full_path)

    return file_name


@app.route('/download/<filename>')
def download_file(filename):
    try:
        return send_file(
            os.path.join('reports', filename),
            as_attachment=True,
            download_name=f"disease_prediction_report.pdf"
        )
    except Exception as e:
        return render_template('error.html', error=f"Download error: {str(e)}")


# Error handler
@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error="Page not found"), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error="Internal server error"), 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
import os
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import joblib
from flask import Flask, jsonify, render_template, request


app = Flask(__name__)


# Load the trained model and scaler
xgb_model = joblib.load("xgb_best_model.pkl")
scaler = joblib.load("scaler.pkl")


# Email credentials must be provided through environment variables.
# Never store real credentials directly in source code.
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")

EMAIL_CONFIGURED = all(
    [SENDER_EMAIL, SENDER_PASSWORD, RECEIVER_EMAIL]
)


def send_alert_email(pH, turbidity, tds, conductivity, confidence):
    """Send an email alert when email credentials are configured."""

    if not EMAIL_CONFIGURED:
        print(
            "Email alert skipped: email environment variables "
            "are not configured."
        )
        return False

    try:
        subject = "Potential Water Pollution Alert"

        body = f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    margin: 0;
                    padding: 0;
                    color: #333;
                }}
                .container {{
                    background-color: #fff;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                }}
                h1 {{
                    color: #d9534f;
                }}
                h3 {{
                    color: #5bc0de;
                }}
                .strong-text {{
                    font-weight: bold;
                }}
                ul {{
                    list-style-type: none;
                    padding: 0;
                }}
                li {{
                    margin-bottom: 10px;
                }}
                .footer {{
                    margin-top: 30px;
                    font-size: 14px;
                    color: #777;
                }}
                .logo {{
                    width: 100px;
                    margin-top: 20px;
                }}
            </style>
        </head>

        <body>
            <div class="container">
                <h1>Potential Water Pollution Alert</h1>

                <p>
                    The prediction model identified a potential plastic-waste
                    risk. Further investigation is recommended.
                </p>

                <h3>Water Quality Parameters</h3>

                <ul>
                    <li>
                        <span class="strong-text">pH Level:</span>
                        {pH}
                    </li>
                    <li>
                        <span class="strong-text">Turbidity:</span>
                        {turbidity} NTU
                    </li>
                    <li>
                        <span class="strong-text">TDS:</span>
                        {tds} mg/L
                    </li>
                    <li>
                        <span class="strong-text">Conductivity:</span>
                        {conductivity} µS/cm
                    </li>
                    <li>
                        <span class="strong-text">Model Confidence:</span>
                        {confidence:.2f}%
                    </li>
                </ul>

                <p>
                    This automated prediction should be verified through
                    appropriate environmental testing before any action is
                    taken.
                </p>

                <div class="footer">
                    <p>PlastiSense Monitoring Prototype</p>
                    <img src="cid:logo" class="logo">
                </div>
            </div>
        </body>
        </html>
        """

        message = MIMEMultipart()
        message["From"] = SENDER_EMAIL
        message["To"] = RECEIVER_EMAIL
        message["Subject"] = subject
        message.attach(MIMEText(body, "html"))

        logo_path = "./static/images/logo.png"

        if os.path.exists(logo_path):
            with open(logo_path, "rb") as image_file:
                image = MIMEImage(image_file.read())
                image.add_header("Content-ID", "<logo>")
                message.attach(image)

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(
                SENDER_EMAIL,
                RECEIVER_EMAIL,
                message.as_string(),
            )

        print("Email alert sent successfully.")
        return True

    except Exception as error:
        print(f"Error sending email alert: {error}")
        return False


@app.route("/")
def home():
    """Display the application home page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Generate a plastic-waste risk prediction."""

    try:
        pH = request.form.get("pH")
        turbidity = request.form.get("Turbidity")
        tds = request.form.get("TDS")
        conductivity = request.form.get("Conductivity")

        if not all([pH, turbidity, tds, conductivity]):
            return jsonify(
                {"result": "Error: All fields are required."}
            )

        try:
            features = [
                float(pH),
                float(turbidity),
                float(tds),
                float(conductivity),
            ]
        except ValueError:
            return jsonify(
                {"result": "Error: Please provide valid numeric values."}
            )

        scaled_features = scaler.transform([features])

        predicted_class = int(
            xgb_model.predict(scaled_features)[0]
        )

        class_probabilities = xgb_model.predict_proba(
            scaled_features
        )[0]

        confidence = float(
            class_probabilities[predicted_class] * 100
        )

        if predicted_class == 0:
            result = "No plastic-waste risk detected."
            alert_status = ""

        else:
            result = (
                "Potential plastic-waste risk detected. "
                "Further investigation is recommended."
            )

            if not EMAIL_CONFIGURED:
                alert_status = (
                    "Email alert is not configured for this demo."
                )
            elif send_alert_email(
                pH,
                turbidity,
                tds,
                conductivity,
                confidence,
            ):
                alert_status = "Email alert sent successfully."
            else:
                alert_status = "Email alert could not be sent."

        return jsonify(
            {
                "result": result,
                "alert_status": alert_status,
                "confidence": f"{confidence:.2f}%",
            }
        )

    except Exception as error:
        print(f"Prediction error: {error}")

        return jsonify(
            {
                "result": (
                    "An unexpected error occurred. "
                    "Please try again later."
                )
            }
        )


if __name__ == "__main__":
    app.run(debug=False)

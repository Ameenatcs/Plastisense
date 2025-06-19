import pandas as pd
import pickle
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request, render_template
import joblib
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email import encoders

app = Flask(__name__)

# Load the Model
xgb_model = joblib.load('xgb_best_model.pkl')  
scaler = joblib.load('scaler.pkl')  

# Email settings
SENDER_EMAIL = 'ethicspractical@gmail.com'
SENDER_PASSWORD = 'lfpi bzns nonq cpvu'  
RECEIVER_EMAIL = 'swnauthorities@gmail.com'

# Function to send email alert
def send_alert_email(pH, turbidity, tds, conductivity, confidence):
    try:
        subject = "🚨 Urgent: Water Pollution Alert"
        
        # Email Body with CID for inline image
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
                <h1>🚨 Urgent: Water Pollution Alert</h1>
                <p>Dear Authorities,</p>
                <p>A water pollution risk has been detected. Immediate action is required!</p>
                
                <h3>Water Quality Parameters:</h3>
                <ul>
                    <li><span class="strong-text">pH Level:</span> {pH}</li>
                    <li><span class="strong-text">Turbidity:</span> {turbidity} NTU</li>
                    <li><span class="strong-text">TDS:</span> {tds} mg/L</li>
                    <li><span class="strong-text">Conductivity:</span> {conductivity} µS/cm</li>
                    <li><span class="strong-text">Confidence Level:</span> <strong>{confidence:.2f}%</strong></li>
                </ul>

                <p><strong>⚠ Potential plastic waste contamination detected in the water source.</strong></p>
                <p><strong>Please investigate immediately to prevent environmental damage.</strong></p>

                <div class="footer">
                    <p>Best Regards,</p>
                    <img src="cid:logo" class="logo">
                </div>
            </div>
        </body>
        </html>
        """
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html'))  

        with open("./static/images/logo.png", 'rb') as img_file:
            img = MIMEImage(img_file.read())
            img.add_header('Content-ID', '<logo>') 
            msg.attach(img)

        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())

        print("Email sent successfully!")
        return True

    except Exception as e:
        print(f"Error sending email: {e}")
        return False

# Route for Pages
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        pH = request.form.get('pH')
        turbidity = request.form.get('Turbidity')
        tds = request.form.get('TDS')
        conductivity = request.form.get('Conductivity')

        if not pH or not turbidity or not tds or not conductivity:
            return jsonify({'result': "Error: All fields are required."})
        
        try:
            features = [
                float(pH),
                float(turbidity),
                float(tds),
                float(conductivity)
            ]
        except ValueError:
            return jsonify({'result': "Error: Please provide valid numeric values."})

        # Scale features
        scaled_features = scaler.transform([features])
        # print(scaled_features)
        # Make prediction
        final_pred = xgb_model.predict(scaled_features)[0]
        print(final_pred)
        raw_confidence = max(xgb_model.predict_proba(scaled_features)[0])  
        print(raw_confidence)
        if final_pred == 0:
            # Map 0-1 probability to 0-65%
            confidence = raw_confidence * 65 
            result = "No Plastic Waste Detected."
            alert_status = ""
        else:
            # Map 0-1 probability to 65-100%
            confidence = 65 + (raw_confidence * 35)  
            result = "Plastic Waste Risk Detected! Legal compliance action required."
            alert_status = "Alert message has been sent to authorities." if send_alert_email(pH, turbidity, tds, conductivity, confidence) else "Error: Failed to send alert."

        return jsonify({
            'result': result,
            'alert_status': alert_status,
            'confidence': f"{confidence:.2f}%"
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'result': "An error occurred. Please try again later."})

if __name__ == "__main__":
    app.run(debug=True)

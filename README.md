# PlastiSense

A machine-learning and Flask prototype that estimates potential plastic-waste risk using water-quality indicators.

This academic project was completed as part of the M.Sc. Artificial Intelligence programme at Jönköping University, Sweden.

## Problem

Plastic pollution can affect aquatic environments, but direct laboratory testing and field inspections require time and specialised resources.

PlastiSense explores whether selected water-quality measurements can be used by a machine-learning classifier to indicate potential plastic-waste risk.

The application is an experimental decision-support prototype. It does not directly detect plastic particles and should not replace laboratory analysis or professional environmental assessment.

## Input Features

The model uses four water-quality measurements:

| Feature | Description |
|---|---|
| pH | Acidity or alkalinity of the water |
| Turbidity | Cloudiness caused by suspended particles |
| TDS | Total dissolved solids |
| Conductivity | Ability of the water to conduct electrical current |

## Machine-Learning Pipeline

The training notebook performs the following steps:

1. Load the water-quality dataset
2. Select pH, turbidity, TDS, and conductivity
3. Remove extreme values using the interquartile range
4. Apply Isolation Forest for additional outlier filtering
5. Divide the data into training and testing sets
6. Standardise the input features
7. Train an XGBoost binary classifier
8. Evaluate predictions using accuracy, precision, recall, and F1-score
9. Save the trained model and scaler for use by the Flask application

## Model Evaluation

The recorded test results in the notebook are:

| Metric | Class 0 | Class 1 |
|---|---:|---:|
| Precision | 0.89 | 0.65 |
| Recall | 0.95 | 0.41 |
| F1-score | 0.92 | 0.50 |
| Support | 98,510 | 20,271 |

**Overall test accuracy: 86.09%**

Although the overall accuracy is relatively high, recall for Class 1 is 0.41. This means that the model misses a substantial proportion of positive cases. Further work on class imbalance, threshold selection, and validation is necessary before practical use.

## Web Application

The Flask application:

- Accepts pH, turbidity, TDS, and conductivity values
- Applies the saved feature scaler
- Generates a binary prediction with XGBoost
- Displays the model’s predicted-class probability
- Shows whether potential plastic-waste risk was identified
- Can optionally send an email alert when credentials are configured securely

## Project Structure

```text
Plastisense/
├── Templates/
│   └── index.html
├── static/
│   ├── css/
│   └── images/
├── .gitignore
├── app.py
├── main.ipynb
├── requirements.txt
├── scaler.pkl
└── xgb_best_model.pkl
```

## Run Locally

Clone the repository:

```bash
git clone https://github.com/Ameenatcs/Plastisense.git
cd Plastisense
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the Flask application:

```bash
python app.py
```

Open the application in your browser:

```text
http://127.0.0.1:5000
```

## Optional Email Configuration

Email alerts are disabled unless the required environment variables are configured.

Create a local `.env` file or set the following environment variables in your operating system:

```text
SENDER_EMAIL=your_email@example.com
SENDER_PASSWORD=your_app_password
RECEIVER_EMAIL=recipient@example.com
```

Never commit real email addresses, passwords, API keys, or other credentials to GitHub.

The `.gitignore` file excludes `.env` files from version control.

## Current Limitations

- The model estimates risk from indirect water-quality indicators; it does not directly detect plastic.
- The original training dataset is not included in this repository.
- Class 1 recall is currently low.
- The notebook contains saved outputs from one train–test split.
- The results have not been validated using independent real-world environmental samples.
- Model probabilities should not be interpreted as calibrated risk estimates without additional evaluation.
- Email alerts require external SMTP configuration.

## Future Improvements

- Improve Class 1 recall and address class imbalance
- Compare XGBoost with additional baseline models
- Evaluate precision–recall trade-offs at different thresholds
- Add cross-validation and independent test data
- Calibrate predicted probabilities
- Add automated tests and continuous integration
- Deploy a secure public demonstration
- Validate predictions against laboratory-confirmed measurements

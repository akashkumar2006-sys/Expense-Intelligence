# Expense-Intelligence
# Expense Intelligence

An ML-powered expense tracking and analysis application built with Python, Flask, Pandas, and Scikit-learn.

## Features

- Add and track personal expenses
- Categorize expenses
- Record payment and merchant information
- Analyze spending patterns
- View total, average, and highest expenses
- View category-wise spending
- View monthly spending
- Predict potentially high-risk expenses using Machine Learning
- View expense history
- Delete recorded expenses
- Responsive web interface

## Tech Stack

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Joblib
- HTML
- CSS
- JavaScript

## Machine Learning

The project uses a Random Forest Classifier to classify expenses as:

- Normal
- High Risk

The model is trained using:

data/expense_intelligence_training_data.csv

The trained model and encoder are saved inside:

models/

- expense_risk_model.pkl
- expense_encoder.pkl

## Project Structure

Expense-Intelligence/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── expense_intelligence_training_data.csv
│
├── models/
│   ├── expense_risk_model.pkl
│   └── expense_encoder.pkl
│
├── templates/
│   ├── index.html
│   ├── dashboard.html
│   └── result.html
│
└── static/
    ├── css/
    │   └── style.css
    │
    └── js/
        └── script.js

## Installation

Install the required libraries:

pip install -r requirements.txt

## Train the Model

Before running the application for the first time, train the model:

python train_model.py

This creates the required model files inside the models directory.

## Run the Application

Start Flask:

python app.py

Then open the local Flask address shown in the terminal.

## How It Works

1. User enters expense information.
2. Flask receives the submitted data.
3. The application calculates additional features.
4. The trained ML model analyzes the expense.
5. The application returns a risk classification.
6. The expense is saved to the user's expense history.
7. The dashboard displays spending insights.

## Future Improvements

- Interactive charts
- User authentication
- Advanced spending recommendations
- Monthly budget alerts
- Better model evaluation
- Cloud deployment
- Database integration

## Author

Akash Kumar Jha

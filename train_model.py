import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==============================
# FILE PATHS
# ==============================

DATA_FILE = "data/expense_intelligence_training_data.csv"

MODEL_FOLDER = "models"

MODEL_FILE = os.path.join(
    MODEL_FOLDER,
    "expense_risk_model.pkl"
)

ENCODER_FILE = os.path.join(
    MODEL_FOLDER,
    "expense_encoder.pkl"
)


# ==============================
# CREATE MODEL FOLDER
# ==============================

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)


# ==============================
# LOAD DATASET
# ==============================

df = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully.")

print(
    "Dataset shape:",
    df.shape
)


# ==============================
# FEATURES
# ==============================

categorical_columns = [
    "category",
    "payment_method",
    "merchant_type"
]


numerical_columns = [
    "age",
    "monthly_income",
    "amount",
    "days_since_last_expense",
    "monthly_expense_count",
    "budget_limit",
    "amount_to_budget_ratio",
    "income_expense_ratio"
]


feature_columns = (
    numerical_columns +
    categorical_columns
)


X = df[feature_columns]

y = df["high_risk_expense"]


# ==============================
# ENCODE CATEGORICAL DATA
# ==============================

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)


encoded_data = encoder.fit_transform(
    X[categorical_columns]
)


# ==============================
# NUMERICAL DATA
# ==============================

numerical_data = X[
    numerical_columns
].values


# ==============================
# COMBINE FEATURES
# ==============================

import numpy as np


final_X = np.hstack([
    numerical_data,
    encoded_data
])


# ==============================
# TRAIN / TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(

    final_X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)


# ==============================
# CREATE MODEL
# ==============================

model = RandomForestClassifier(

    n_estimators=200,

    random_state=42,

    class_weight="balanced"

)


# ==============================
# TRAIN
# ==============================

print("Training model...")

model.fit(
    X_train,
    y_train
)


# ==============================
# TEST MODEL
# ==============================

predictions = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    predictions
)


print()
print(
    "Model Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print()

print(
    classification_report(
        y_test,
        predictions
    )
)


# ==============================
# SAVE MODEL
# ==============================

joblib.dump(
    model,
    MODEL_FILE
)


joblib.dump(
    encoder,
    ENCODER_FILE
)


print()
print("Model saved successfully.")

print(
    "Model:",
    MODEL_FILE
)

print(
    "Encoder:",
    ENCODER_FILE
)

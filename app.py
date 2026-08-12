from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os
import joblib


app = Flask(__name__)


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
# LOAD MODEL
# ==============================

model = None
encoder = None

if os.path.exists(MODEL_FILE):

    model = joblib.load(MODEL_FILE)

if os.path.exists(ENCODER_FILE):

    encoder = joblib.load(ENCODER_FILE)


# ==============================
# HOME
# ==============================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==============================
# ADD / ANALYZE EXPENSE
# ==============================

@app.route(
    "/add-expense",
    methods=["POST"]
)
def add_expense():

    try:

        date = request.form.get("date")
        item = request.form.get("item")
        amount = float(
            request.form.get("amount")
        )

        category = request.form.get(
            "category"
        )

        payment_method = request.form.get(
            "payment_method"
        )

        merchant_type = request.form.get(
            "merchant_type"
        )

        age = int(
            request.form.get("age")
        )

        monthly_income = float(
            request.form.get(
                "monthly_income"
            )
        )

        budget_limit = float(
            request.form.get(
                "budget_limit"
            )
        )

        days_since_last_expense = int(
            request.form.get(
                "days_since_last_expense"
            )
        )

        monthly_expense_count = int(
            request.form.get(
                "monthly_expense_count"
            )
        )


        # ==============================
        # VALIDATION
        # ==============================

        if amount <= 0:
            raise ValueError(
                "Amount must be greater than zero."
            )

        if monthly_income <= 0:
            raise ValueError(
                "Monthly income must be greater than zero."
            )

        if budget_limit <= 0:
            raise ValueError(
                "Budget limit must be greater than zero."
            )


        # ==============================
        # DERIVED FEATURES
        # ==============================

        amount_to_budget_ratio = (
            amount / budget_limit
        )

        income_expense_ratio = (
            amount / monthly_income
        )


        # ==============================
        # MODEL PREDICTION
        # ==============================

        risk_level = "Normal"
        risk_probability = 0.0


        if model is not None and encoder is not None:

            input_data = pd.DataFrame([{

                "age": age,

                "monthly_income":
                    monthly_income,

                "category":
                    category,

                "amount":
                    amount,

                "payment_method":
                    payment_method,

                "merchant_type":
                    merchant_type,

                "days_since_last_expense":
                    days_since_last_expense,

                "monthly_expense_count":
                    monthly_expense_count,

                "budget_limit":
                    budget_limit,

                "amount_to_budget_ratio":
                    amount_to_budget_ratio,

                "income_expense_ratio":
                    income_expense_ratio

            }])


            categorical_columns = [
                "category",
                "payment_method",
                "merchant_type"
            ]


            encoded = encoder.transform(
                input_data[
                    categorical_columns
                ]
            )


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


            numerical_data = input_data[
                numerical_columns
            ].values


            import numpy as np

            final_input = np.hstack([
                numerical_data,
                encoded
            ])


            prediction = model.predict(
                final_input
            )[0]


            if hasattr(
                model,
                "predict_proba"
            ):

                probability = model.predict_proba(
                    final_input
                )[0][1]

                risk_probability = (
                    probability * 100
                )


            if prediction == 1:

                risk_level = "High Risk"

            else:

                risk_level = "Normal"


        # ==============================
        # SAVE EXPENSE
        # ==============================

        expense_record = pd.DataFrame([{

            "date": date,

            "item": item,

            "amount": amount,

            "category": category,

            "payment_method":
                payment_method,

            "merchant_type":
                merchant_type,

            "age": age,

            "monthly_income":
                monthly_income,

            "budget_limit":
                budget_limit,

            "days_since_last_expense":
                days_since_last_expense,

            "monthly_expense_count":
                monthly_expense_count,

            "amount_to_budget_ratio":
                amount_to_budget_ratio,

            "income_expense_ratio":
                income_expense_ratio,

            "risk_level":
                risk_level,

            "risk_probability":
                risk_probability

        }])


        # Save separately as user expense history

        user_file = (
            "data/user_expenses.csv"
        )


        if os.path.exists(user_file):

            old_data = pd.read_csv(
                user_file
            )

            expense_record = pd.concat(
                [
                    old_data,
                    expense_record
                ],
                ignore_index=True
            )


        expense_record.to_csv(
            user_file,
            index=False
        )


        # ==============================
        # RESULT PAGE
        # ==============================

        return render_template(

            "result.html",

            success=True,

            expense={
                "date": date,
                "item": item,
                "amount": amount,
                "category": category,
                "payment_method":
                    payment_method
            },

            risk_level=risk_level,

            risk_probability=
                risk_probability

        )


    except Exception as e:

        return render_template(

            "result.html",

            success=False,

            message=str(e)

        )


# ==============================
# DASHBOARD
# ==============================

@app.route("/dashboard")
def dashboard():

    user_file = (
        "data/user_expenses.csv"
    )


    if not os.path.exists(
        user_file
    ):

        return render_template(

            "dashboard.html",

            expenses=[],

            total=0,

            average=0,

            highest=0,

            high_risk_count=0,

            category_summary={},

            monthly_summary={}

        )


    df = pd.read_csv(
        user_file
    )


    if df.empty:

        return render_template(

            "dashboard.html",

            expenses=[],

            total=0,

            average=0,

            highest=0,

            high_risk_count=0,

            category_summary={},

            monthly_summary={}

        )


    # ==============================
    # NUMERIC CONVERSION
    # ==============================

    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce"
    ).fillna(0)


    # ==============================
    # BASIC STATISTICS
    # ==============================

    total = df["amount"].sum()

    average = df["amount"].mean()

    highest = df["amount"].max()


    # ==============================
    # HIGH RISK COUNT
    # ==============================

    high_risk_count = int(
        (
            df["risk_level"]
            == "High Risk"
        ).sum()
    )


    # ==============================
    # CATEGORY ANALYSIS
    # ==============================

    category_summary = (

        df.groupby("category")["amount"]

        .sum()

        .sort_values(
            ascending=False
        )

        .to_dict()

    )


    # ==============================
    # MONTHLY ANALYSIS
    # ==============================

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )


    df["month"] = (
        df["date"]
        .dt.strftime("%Y-%m")
    )


    monthly_summary = (

        df.dropna(
            subset=["month"]
        )

        .groupby("month")["amount"]

        .sum()

        .sort_index()

        .to_dict()

    )


    # ==============================
    # FORMAT DATE
    # ==============================

    df["date"] = (
        df["date"]
        .dt.strftime("%Y-%m-%d")
    )


    # ==============================
    # SEND DATA TO HTML
    # ==============================

    expenses = df.to_dict(
        orient="records"
    )


    return render_template(

        "dashboard.html",

        expenses=expenses,

        total=total,

        average=average,

        highest=highest,

        high_risk_count=
            high_risk_count,

        category_summary=
            category_summary,

        monthly_summary=
            monthly_summary

    )


# ==============================
# DELETE EXPENSE
# ==============================

@app.route(
    "/delete-expense/<int:index>",
    methods=["POST"]
)
def delete_expense(index):

    user_file = (
        "data/user_expenses.csv"
    )


    if os.path.exists(
        user_file
    ):

        df = pd.read_csv(
            user_file
        )


        if 0 <= index < len(df):

            df = df.drop(
                index
            ).reset_index(
                drop=True
            )


            df.to_csv(
                user_file,
                index=False
            )


    return redirect(
        url_for("dashboard")
    )


# ==============================
# RUN APP
# ==============================

if __name__ == "__main__":

    app.run(
        debug=True
  )

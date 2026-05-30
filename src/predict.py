import joblib


def predict_sales(advertising_amount):

    model = joblib.load(
        "models/linear_regression_model.pkl"
    )

    prediction = model.predict(
        [[advertising_amount]]
    )

    return prediction[0]
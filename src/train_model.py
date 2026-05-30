from sklearn.linear_model import LinearRegression
import joblib
import os


def train_model(X_train, y_train):

    model = LinearRegression()

    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        model,
        "models/linear_regression_model.pkl"
    )

    print("\nModel Trained Successfully")

    print(
        "\nModel Coefficient:",
        model.coef_[0]
    )

    print(
        "Model Intercept:",
        model.intercept_
    )

    return model
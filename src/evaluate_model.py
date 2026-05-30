from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import matplotlib.pyplot as plt
import numpy as np
import os


def evaluate_model(
        model,
        X_train,
        X_test,
        y_train,
        y_test
):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions
    )

    print("\n")
    print("=" * 50)

    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

    print("=" * 50)

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    # Scatter Plot

    plt.figure(figsize=(8, 5))

    plt.scatter(
        X_train,
        y_train
    )

    plt.title(
        "Advertising vs Sales"
    )

    plt.xlabel(
        "TV Advertising"
    )

    plt.ylabel(
        "Sales"
    )

    plt.savefig(
        "outputs/scatter_plot.png"
    )

    plt.close()

    # Regression Line

    plt.figure(figsize=(8, 5))

    plt.scatter(
        X_train,
        y_train
    )

    plt.plot(
        X_train,
        model.predict(X_train)
    )

    plt.title(
        "Regression Line"
    )

    plt.xlabel(
        "TV Advertising"
    )

    plt.ylabel(
        "Sales"
    )

    plt.savefig(
        "outputs/regression_line.png"
    )

    plt.close()

    # Actual vs Predicted

    plt.figure(figsize=(8, 5))

    plt.scatter(
        y_test,
        predictions
    )

    plt.xlabel(
        "Actual Sales"
    )

    plt.ylabel(
        "Predicted Sales"
    )

    plt.title(
        "Actual vs Predicted Sales"
    )

    plt.savefig(
        "outputs/prediction_comparison.png"
    )

    plt.close()

    print("\nGraphs Saved Successfully")

    print(
        "outputs/scatter_plot.png"
    )

    print(
        "outputs/regression_line.png"
    )

    print(
        "outputs/prediction_comparison.png"
    )
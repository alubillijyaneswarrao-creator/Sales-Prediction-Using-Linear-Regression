from src.data_preprocessing import (
    load_and_preprocess_data
)

from src.train_model import (
    train_model
)

from src.evaluate_model import (
    evaluate_model
)


def main():

    dataset_path = (
        "dataset/advertising.csv"
    )

    X_train, X_test, y_train, y_test = (
        load_and_preprocess_data(
            dataset_path
        )
    )

    model = train_model(
        X_train,
        y_train
    )

    evaluate_model(
        model,
        X_train,
        X_test,
        y_train,
        y_test
    )


if __name__ == "__main__":
    main()
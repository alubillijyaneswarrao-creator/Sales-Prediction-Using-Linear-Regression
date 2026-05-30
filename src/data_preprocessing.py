import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(filepath):

    df = pd.read_csv(filepath)

    print("\nDataset Shape:")
    print(df.shape)

    print("\nFirst 5 Records:")
    print(df.head())

    X = df[['TV']]
    y = df['Sales']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("\nTraining Samples:", len(X_train))
    print("Testing Samples:", len(X_test))

    return X_train, X_test, y_train, y_test
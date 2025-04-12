import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    # Check if 'target' column is present
    if 'target' not in df.columns:
        raise ValueError("CSV file must contain a 'target' column.")

    # Replace '?' or missing with NaN
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)

    # Convert target to binary (1 if > 0 else 0)
    df['target'] = df['target'].apply(lambda x: 1 if int(x) > 0 else 0)

    return df


def split_features_target(data):
    """
    Split features and target variable.
    """
    X = data.drop("target", axis=1)
    y = data["target"]
    return X, y

def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

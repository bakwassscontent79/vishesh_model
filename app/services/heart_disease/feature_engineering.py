import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

def select_best_features(X_train, y_train, X_test, k=10):
    """
    Select top k features using ANOVA F-value.
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    selected_indices = selector.get_support(indices=True)
    if isinstance(X_train, pd.DataFrame):
        selected_features = X_train.columns[selected_indices]
        print(f"Selected Features: {list(selected_features)}")

    return X_train_selected, X_test_selected

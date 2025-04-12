import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

def train_models(X_train, y_train):
    """
    Train multiple classification models.
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

    return models

def save_models(models, output_dir='app/static/models/'):
    """
    Save trained models to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    for name, model in models.items():
        filename = name.lower().replace(" ", "_") + ".pkl"
        filepath = os.path.join(output_dir, filename)
        joblib.dump(model, filepath)
        print(f"Saved: {filepath}")

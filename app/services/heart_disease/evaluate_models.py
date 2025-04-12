import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

def evaluate_models(models, X_test, y_test):
    """
    Evaluate multiple models on test data.

    Parameters:
    -----------
    models : dict
        Dictionary of trained models.
    X_test : array-like
        Testing features.
    y_test : array-like
        Testing target.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing evaluation metrics for each model.
    """
    results = {}

    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Store the results
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        # Print the results
        print(f"\n{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("-" * 50)
        print(f"Classification Report:")
        print(classification_report(y_test, y_pred))

    # Convert results to DataFrame
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[model]['accuracy'] for model in results],
        'Precision': [results[model]['precision'] for model in results],
        'Recall': [results[model]['recall'] for model in results],
        'F1 Score': [results[model]['f1_score'] for model in results]
    })

    return results_df

def get_confusion_matrices(models, X_test, y_test):
    """
    Get confusion matrices for multiple models.

    Parameters:
    -----------
    models : dict
        Dictionary of trained models.
    X_test : array-like
        Testing features.
    y_test : array-like
        Testing target.

    Returns:
    --------
    dict
        Dictionary containing confusion matrices for each model.
    """
    cm_results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        cm_results[name] = cm

    return cm_results

def get_roc_curves(models, X_test, y_test):
    """
    Calculate ROC curves for multiple models.

    Parameters:
    -----------
    models : dict
        Dictionary of trained models.
    X_test : array-like
        Testing features.
    y_test : array-like
        Testing target.

    Returns:
    --------
    dict
        Dictionary containing ROC curve data (fpr, tpr, auc) for each model.
    """
    roc_results = {}

    for name, model in models.items():
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_proba = model.decision_function(X_test)

            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)

            roc_results[name] = {
                'fpr': fpr,
                'tpr': tpr,
                'auc': roc_auc
            }
        except Exception as e:
            print(f"Error calculating ROC curve for {name}: {e}")

    return roc_results

def compare_models_performance(initial_results, tuned_results):
    """
    Compare the performance of models before and after tuning.

    Parameters:
    -----------
    initial_results : pd.DataFrame
        Results of models before tuning.
    tuned_results : pd.DataFrame
        Results of models after tuning.

    Returns:
    --------
    pd.DataFrame
        DataFrame showing the improvement in performance.
    """
    comparison_df = pd.DataFrame({
        'Model': initial_results['Model'],
        'Initial Accuracy': initial_results['Accuracy'],
        'Tuned Accuracy': tuned_results['Accuracy'],
        'Accuracy Improvement': tuned_results['Accuracy'] - initial_results['Accuracy'],
        'Initial F1': initial_results['F1 Score'],
        'Tuned F1': tuned_results['F1 Score'],
        'F1 Improvement': tuned_results['F1 Score'] - initial_results['F1 Score']
    })

    return comparison_df

# ✅ NEW HOME ROUTE FIX
from flask import redirect, url_for
from app.routes.admin_routes import admin_routes

@admin_routes.route("/")
def home():
    return redirect(url_for("admin_routes.heart_ui"))

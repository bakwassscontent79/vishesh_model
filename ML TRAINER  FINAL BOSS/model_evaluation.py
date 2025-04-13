import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
)
import seaborn as sns

def calculate_metrics(model, X_train, X_test, y_train, y_test):
    """
    Calculate various performance metrics for a model.
    
    Args:
        model: The trained model.
        X_train: Training features.
        X_test: Test features.
        y_train: Training target.
        y_test: Test target.
        
    Returns:
        dict: A dictionary of performance metrics.
    """
    # Make predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Calculate probabilities if the model supports it
    train_proba = None
    test_proba = None
    
    try:
        train_proba = model.predict_proba(X_train)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]
    except (AttributeError, IndexError):
        # Some models might not support predict_proba
        pass
    
    # Calculate basic metrics
    metrics = {
        'train_accuracy': accuracy_score(y_train, train_preds),
        'test_accuracy': accuracy_score(y_test, test_preds),
        'train_precision': precision_score(y_train, train_preds, zero_division=0),
        'test_precision': precision_score(y_test, test_preds, zero_division=0),
        'train_recall': recall_score(y_train, train_preds),
        'test_recall': recall_score(y_test, test_preds),
        'train_f1': f1_score(y_train, train_preds),
        'test_f1': f1_score(y_test, test_preds),
        'train_confusion_matrix': confusion_matrix(y_train, train_preds),
        'test_confusion_matrix': confusion_matrix(y_test, test_preds)
    }
    
    # Calculate ROC curve and AUC if probabilities are available
    if train_proba is not None and test_proba is not None:
        # ROC curve
        train_fpr, train_tpr, _ = roc_curve(y_train, train_proba)
        test_fpr, test_tpr, _ = roc_curve(y_test, test_proba)
        
        metrics['train_fpr'] = train_fpr
        metrics['train_tpr'] = train_tpr
        metrics['test_fpr'] = test_fpr
        metrics['test_tpr'] = test_tpr
        metrics['train_roc_auc'] = auc(train_fpr, train_tpr)
        metrics['test_roc_auc'] = auc(test_fpr, test_tpr)
        
        # Precision-Recall curve
        train_precision_curve, train_recall_curve, _ = precision_recall_curve(y_train, train_proba)
        test_precision_curve, test_recall_curve, _ = precision_recall_curve(y_test, test_proba)
        
        metrics['train_precision_curve'] = train_precision_curve
        metrics['train_recall_curve'] = train_recall_curve
        metrics['test_precision_curve'] = test_precision_curve
        metrics['test_recall_curve'] = test_recall_curve
        metrics['train_average_precision'] = average_precision_score(y_train, train_proba)
        metrics['test_average_precision'] = average_precision_score(y_test, test_proba)
    else:
        # Set default values if probabilities are not available
        metrics['train_roc_auc'] = 0.5
        metrics['test_roc_auc'] = 0.5
        metrics['train_average_precision'] = 0.0
        metrics['test_average_precision'] = 0.0
    
    return metrics

def evaluate_models(models, X_train, X_test, y_train, y_test):
    """
    Evaluate multiple models and calculate their performance metrics.
    
    Args:
        models (dict): Dictionary of trained models.
        X_train: Training features.
        X_test: Test features.
        y_train: Training target.
        y_test: Test target.
        
    Returns:
        dict: Dictionary of performance metrics for each model.
    """
    results = {}
    
    for model_name, model in models.items():
        # Calculate metrics for the model
        metrics = calculate_metrics(model, X_train, X_test, y_train, y_test)
        results[model_name] = metrics
    
    return results

def plot_confusion_matrix(cm, title='Confusion Matrix'):
    """
    Plot a confusion matrix.
    
    Args:
        cm (numpy.ndarray): Confusion matrix.
        title (str, optional): Title for the plot. Defaults to 'Confusion Matrix'.
        
    Returns:
        matplotlib.figure.Figure: The figure with the confusion matrix.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        square=True,
        cbar=False,
        ax=ax
    )
    
    # Set labels and title
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    
    # Set tick labels
    ax.set_xticklabels(['No Disease', 'Disease'])
    ax.set_yticklabels(['No Disease', 'Disease'])
    
    plt.tight_layout()
    return fig

def plot_roc_curve(fpr, tpr, roc_auc):
    """
    Plot a ROC curve.
    
    Args:
        fpr (numpy.ndarray): False positive rates.
        tpr (numpy.ndarray): True positive rates.
        roc_auc (float): Area under the ROC curve.
        
    Returns:
        matplotlib.figure.Figure: The figure with the ROC curve.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Plot ROC curve
    ax.plot(
        fpr, tpr, 
        color='darkorange',
        lw=2, 
        label=f'ROC curve (AUC = {roc_auc:.3f})'
    )
    
    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    # Set labels and title
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    
    # Set limits and grid
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    return fig

def plot_precision_recall_curve(precision, recall, average_precision):
    """
    Plot a precision-recall curve.
    
    Args:
        precision (numpy.ndarray): Precision values.
        recall (numpy.ndarray): Recall values.
        average_precision (float): Average precision score.
        
    Returns:
        matplotlib.figure.Figure: The figure with the precision-recall curve.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Plot precision-recall curve
    ax.plot(
        recall, precision, 
        color='blue',
        lw=2, 
        label=f'AP = {average_precision:.3f}'
    )
    
    # Set labels and title
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    
    # Set limits and grid
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc='best')
    
    plt.tight_layout()
    return fig

def plot_feature_importance(model, feature_names, model_name='Model'):
    """
    Plot feature importance for a model.
    
    Args:
        model: The trained model.
        feature_names (list): List of feature names.
        model_name (str, optional): Name of the model. Defaults to 'Model'.
        
    Returns:
        matplotlib.figure.Figure: The figure with the feature importance plot.
    """
    # Get feature importance
    importance = None
    
    # Different models have different ways to access feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        raise AttributeError(f"Model {model_name} doesn't have feature importance attributes.")
    
    # Sort features by importance
    indices = np.argsort(importance)[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_importance = importance[indices]
    
    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(range(len(sorted_importance)), sorted_importance, align='center')
    ax.set_xticks(range(len(sorted_importance)))
    ax.set_xticklabels(sorted_names, rotation=45, ha='right')
    
    ax.set_title(f'Feature Importance - {model_name}')
    ax.set_ylabel('Importance')
    ax.set_xlabel('Feature')
    
    plt.tight_layout()
    return fig

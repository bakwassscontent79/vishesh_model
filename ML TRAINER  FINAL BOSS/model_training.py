import numpy as np
import pandas as pd
import time
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

def get_default_hyperparameters(model_name):
    """
    Get default hyperparameters for a specific model.
    
    Args:
        model_name (str): The name of the model.
        
    Returns:
        dict: Default hyperparameters for the model.
    """
    default_params = {
        "Logistic Regression": {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 100
        },
        "Decision Tree": {
            "max_depth": 5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "criterion": "gini"
        },
        "Random Forest": {
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "criterion": "gini"
        },
        "SVM": {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale"
        },
        "Gradient Boosting": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "subsample": 1.0
        },
        "KNN": {
            "n_neighbors": 5,
            "weights": "uniform",
            "algorithm": "auto",
            "p": 2
        }
    }
    
    return default_params.get(model_name, {})

def get_hyperparameter_options(model_name):
    """
    Get hyperparameter options for a specific model.
    
    Args:
        model_name (str): The name of the model.
        
    Returns:
        dict: Hyperparameter options for the model.
    """
    param_options = {
        "Logistic Regression": {
            "C": (0.01, 10.0, 0.01),
            "penalty": ["l1", "l2", "elasticnet", "none"],
            "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            "max_iter": (50, 1000, 50)
        },
        "Decision Tree": {
            "max_depth": (1, 20, 1),
            "min_samples_split": (2, 20, 1),
            "min_samples_leaf": (1, 20, 1),
            "criterion": ["gini", "entropy", "log_loss"]
        },
        "Random Forest": {
            "n_estimators": (10, 500, 10),
            "max_depth": (1, 20, 1),
            "min_samples_split": (2, 20, 1),
            "min_samples_leaf": (1, 10, 1),
            "criterion": ["gini", "entropy", "log_loss"]
        },
        "SVM": {
            "C": (0.01, 10.0, 0.01),
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "gamma": ["scale", "auto"]
        },
        "Gradient Boosting": {
            "n_estimators": (10, 500, 10),
            "learning_rate": (0.01, 1.0, 0.01),
            "max_depth": (1, 10, 1),
            "min_samples_split": (2, 20, 1),
            "min_samples_leaf": (1, 10, 1),
            "subsample": (0.1, 1.0, 0.1)
        },
        "KNN": {
            "n_neighbors": (1, 20, 1),
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "p": [1, 2]
        }
    }
    
    return param_options.get(model_name, {})

def get_model(model_name, params=None):
    """
    Get a model instance based on the model name.
    
    Args:
        model_name (str): The name of the model.
        params (dict, optional): Parameters for the model. Defaults to None.
        
    Returns:
        object: A model instance.
    """
    if params is None:
        params = {}
    
    models = {
        "Logistic Regression": LogisticRegression,
        "Decision Tree": DecisionTreeClassifier,
        "Random Forest": RandomForestClassifier,
        "SVM": SVC,
        "Gradient Boosting": GradientBoostingClassifier,
        "KNN": KNeighborsClassifier
    }
    
    model_class = models.get(model_name)
    if model_class:
        return model_class(**params)
    
    raise ValueError(f"Model {model_name} not supported.")

def train_models(X_train, y_train, selected_models):
    """
    Train multiple models on the training data.
    
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target.
        selected_models (list): List of model names to train.
        
    Returns:
        tuple: (models, train_times)
            - models (dict): Dictionary of trained models.
            - train_times (dict): Dictionary of training times.
    """
    models = {}
    train_times = {}
    
    for model_name in selected_models:
        # Get default params for the model
        default_params = get_default_hyperparameters(model_name)
        
        # Get the model instance
        model = get_model(model_name, default_params)
        
        # Train the model and measure time
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Store the model and training time
        models[model_name] = model
        train_times[model_name] = train_time
    
    return models, train_times

def hyperparameter_tuning(model_name, params, X_train, y_train, cv=5):
    """
    Perform hyperparameter tuning for a model.
    
    Args:
        model_name (str): The name of the model.
        params (dict): Parameters to tune.
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target.
        cv (int, optional): Number of cross-validation folds. Defaults to 5.
        
    Returns:
        tuple: (best_model, cv_results)
            - best_model (object): The best trained model.
            - cv_results (pandas.DataFrame): Cross-validation results.
    """
    # Create a parameter grid with a single combination (the provided params)
    param_grid = {param: [value] for param, value in params.items()}
    
    # Get the model instance
    base_model = get_model(model_name, {})
    
    # Create a grid search object
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        return_train_score=True,
        verbose=1
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Return the best model and CV results
    return best_model, pd.DataFrame(grid_search.cv_results_)

import os
import uuid
import pandas as pd
import plotly
import json

from flask import Blueprint, render_template, request, send_file, redirect, url_for
from app.services.heart_disease.data_processor import preprocess_data
from app.services.heart_disease.feature_engineering import select_best_features
from app.services.heart_disease.model_trainer import train_models
from app.services.heart_disease.evaluate_models import (
    evaluate_models,
    get_confusion_matrices,
    get_roc_curves,
)

# ✅ Set correct route prefix for Blueprint
heart_disease_bp = Blueprint("heart_disease", __name__, url_prefix='/admin/heart-disease')

UPLOAD_FOLDER = "app/static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



@heart_disease_bp.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file part", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    filepath = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.csv")
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)
        df = preprocess_data(df)
    except Exception as e:
        return f"Error processing file: {e}", 400

    # Proceed if all is good
    y = df["target"]
    X = df.drop("target", axis=1)
    X_train, X_test = X[:int(0.8 * len(X))], X[int(0.8 * len(X)):]
    y_train, y_test = y[:int(0.8 * len(y))], y[int(0.8 * len(y)):]

    X_train_selected, X_test_selected = select_best_features(X_train, y_train, X_test)

    models = train_models(X_train_selected, y_train)
    results_df = evaluate_models(models, X_test_selected, y_test)
    roc_data = get_roc_curves(models, X_test_selected, y_test)

    roc_json = json.dumps([
        {
            "name": name,
            "fpr": list(data["fpr"]),
            "tpr": list(data["tpr"]),
            "auc": data["auc"]
        }
        for name, data in roc_data.items()
    ])

    return render_template(
        "heart_disease.html",
        tables=[results_df.to_html(classes='data table table-bordered', header=True, index=False)],
        roc_data=roc_json,
        csv_path=filepath
    )




@heart_disease_bp.route("/download-csv", methods=["GET"])
def download_csv():
    path = request.args.get("path")
    return send_file(path, as_attachment=True)

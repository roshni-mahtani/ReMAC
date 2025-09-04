import os
import sys
sys.path.append('../')
import json
import numpy as np
import pandas as pd
import joblib

from utils.load_preprocess import load_data, preprocess_data
from utils.metrics import calculate_metrics, save_validation_summary, save_epoch_metrics_log, calculate_global_metrics
from utils.visualization_utils import plot_ml_results
from utils.class_weights import compute_pos_weight_for_xgb
from utils.random_seed import set_seed

from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

set_seed(5)

def get_model(model_name, params, y_train=None):
    if model_name == "logistic":
        return LogisticRegression(class_weight="balanced", **params)
    elif model_name == "knn":
        return KNeighborsClassifier(**params)
    elif model_name == "svm":
        base_svm = SVC(class_weight="balanced", probability=False, **params)
        # Envolver con calibración para obtener probabilidades calibradas
        calibrated_svm = CalibratedClassifierCV(base_svm, method='sigmoid', cv=4)
        return calibrated_svm
    elif model_name == "random_forest":
        return RandomForestClassifier(class_weight="balanced", **params)
    elif model_name == "xgboost":
        if y_train is not None:
            scale_pos_weight = compute_pos_weight_for_xgb(y_train)
            return XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric='logloss', **params)
        else:
            return XGBClassifier(eval_metric='logloss', **params)
    else:
        raise ValueError("Unsupported model name")

def train_and_evaluate(model_name, model_params, dataset_name, n_folds, scaling_method, X, y, output_dir=None):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=5)
    all_fold_metrics = []
    all_preds = []
    all_targets = []

    if output_dir:
        figures_dir = os.path.join(output_dir, "figures")
        metrics_dir = os.path.join(output_dir, "metrics")
        weights_dir = os.path.join(output_dir, "weights")
        predictions_dir = os.path.join(output_dir, "predictions")
        os.makedirs(figures_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(weights_dir, exist_ok=True)
        os.makedirs(predictions_dir, exist_ok=True)


    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if dataset_name != 'latent':
            X_train_scaled, X_val_scaled = preprocess_data(X_train, X_val, scaling_method)
        else:
            X_train_scaled, X_val_scaled = X_train, X_val

        # Crear y entrenar el modelo
        model = get_model(model_name, model_params, y_train=y_train)
        model.fit(X_train_scaled, y_train)

        # Guardar el modelo
        model_filename = os.path.join(weights_dir, f"best_model_fold{fold+1}.pkl")
        joblib.dump(model, model_filename)

        # Evaluar el modelo
        probs = model.predict_proba(X_val_scaled)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_val_scaled)
        fold_metrics = calculate_metrics(y_val, probs)

        # Guardar visualizaciones
        if output_dir:
            fig_path = os.path.join(figures_dir, f"fold_{fold+1}_plots.png")
            plot_ml_results(fold, y_val, probs, fig_path)

            # Guardar JSON de métricas por fold
            save_epoch_metrics_log(fold_metrics, fold + 1, metrics_dir)

            # Guardar las predicciones y los targets
            np.save(os.path.join(predictions_dir, f"val_preds_fold{fold+1}.npy"), probs)
            np.save(os.path.join(predictions_dir, f"val_targets_fold{fold+1}.npy"), y_val)


        all_fold_metrics.append(fold_metrics)
        all_preds.extend(probs)
        all_targets.extend(y_val)

    # Guardar resumen final (media + std)
    if output_dir:
        global_metrics = calculate_global_metrics(all_targets, all_preds, thresholds=0.5)
        save_validation_summary(all_fold_metrics, output_dir=f"{output_dir}/metrics", global_metrics=global_metrics)

    # Calcular métricas promedio
    avg_metrics = {k: np.mean([fm[k] for fm in all_fold_metrics]) for k in all_fold_metrics[0].keys()}

    return avg_metrics

def run_ml_training(model_name, model_params, dataset, n_folds, scaling_method, output_dir):
    # Cargar los datos
    X, y = load_data(dataset)

    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)

    # Guardar parámetros del modelo
    params_data = {
        "model_name": model_name,
        "model_params": model_params,
        "dataset": dataset,
        "n_folds": n_folds,
        "scaling_method": scaling_method
    }
    with open(os.path.join(output_dir, "params.json"), "w") as f:
        json.dump(params_data, f, indent=4)

    # Entrenamiento y evaluación
    avg_metrics = train_and_evaluate(model_name, model_params, dataset, n_folds, scaling_method, X, y, output_dir=output_dir)

    return avg_metrics

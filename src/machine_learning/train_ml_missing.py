import sys
sys.path.append('../')  # Para encontrar carpetas
import os
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
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

set_seed(5)

def get_model(model_name, params, y_train=None):
    if model_name == "catboost":
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        class_weights = [1.0, n_neg / n_pos] if n_pos > 0 else [1.0, 1.0]
        return CatBoostClassifier(**params, class_weights=class_weights, random_seed=5, verbose=0)
    elif model_name == "xgboost":
        if y_train is not None:
            scale_pos_weight = compute_pos_weight_for_xgb(y_train)
            return XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric='logloss', **params)
        else:
            return XGBClassifier(eval_metric='logloss', **params)
    elif model_name == "histgb":
        return HistGradientBoostingClassifier(**params, class_weight='balanced', random_state=5)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

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

        X_train_scaled, X_val_scaled = preprocess_data(X_train, X_val, scaling_method)

        model = get_model(model_name, model_params, y_train=y_train)

        # Balanceo en HistGradientBoosting con sample_weight
        if model_name == "histgb":
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
            model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train_scaled, y_train)

        model_filename = os.path.join(weights_dir, f"best_model_fold{fold+1}.pkl")
        joblib.dump(model, model_filename)

        # Obtener probabilidades de validación
        val_probs = model.predict_proba(X_val_scaled)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_val_scaled)
        fold_metrics = calculate_metrics(y_val, val_probs)

        if output_dir:
            fig_path = os.path.join(figures_dir, f"fold_{fold+1}_plots.png")
            plot_ml_results(fold, y_val, val_probs, fig_path)
            save_epoch_metrics_log(fold_metrics, fold + 1, metrics_dir)
            np.save(os.path.join(predictions_dir, f"val_preds_fold{fold+1}.npy"), val_probs)
            np.save(os.path.join(predictions_dir, f"val_targets_fold{fold+1}.npy"), y_val)

        all_fold_metrics.append(fold_metrics)
        all_preds.extend(val_probs)
        all_targets.extend(y_val)

    if output_dir:
        global_metrics = calculate_global_metrics(all_targets, all_preds, thresholds=0.5)
        save_validation_summary(all_fold_metrics, output_dir=f"{output_dir}/metrics", global_metrics=global_metrics)

    avg_metrics = {k: np.mean([fm[k] for fm in all_fold_metrics]) for k in all_fold_metrics[0].keys()}

    return avg_metrics


def run_ml_training(model_name, model_params, dataset, n_folds, scaling_method, output_dir):
    
    # Cargar datos según dataset
    X, y = load_data(dataset)

    os.makedirs(output_dir, exist_ok=True)

    params_data = {
        "model_name": model_name,
        "model_params": model_params,
        "dataset": dataset,
        "n_folds": n_folds,
        "scaling_method": scaling_method
    }
    with open(os.path.join(output_dir, "params.json"), "w") as f:
        json.dump(params_data, f, indent=4)

    avg_metrics = train_and_evaluate(
        model_name, model_params, dataset, n_folds, scaling_method,
        X, y, output_dir=output_dir)

    return avg_metrics

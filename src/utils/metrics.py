from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import numpy as np
import pandas as pd
import json
import os

def calculate_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    Calcula métricas de evaluación para un clasificador binario.

    Parameters
    ----------
    y_true : array-like
        Etiquetas verdaderas (0 o 1).
    y_pred_probs : array-like
        Probabilidades predichas por el modelo.
    threshold : float, optional
        Umbral para convertir probabilidades en clases (por defecto 0.5).

    Returns
    -------
    dict
        Diccionario con métricas: ACC, SEN, SPE, PPV, NPV, F1, AUC, AUC_PR.
    """
    # Convertir probabilidades en predicciones binarias
    y_pred = (y_pred_probs >= threshold).astype(int)

    # Matriz de confusión: TN, FP, FN, TP
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    # Cálculo de métricas
    acc = accuracy_score(y_true, y_pred)
    sen = recall_score(y_true, y_pred, zero_division=1)  # Sensibilidad (TPR)
    spe = TN / (TN + FP) if (TN + FP) > 0 else 0         # Especificidad (TNR)
    ppv = precision_score(y_true, y_pred, zero_division=1)  # Valor predictivo positivo
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0          # Valor predictivo negativo
    f1 = f1_score(y_true, y_pred, zero_division=1)
    auc = roc_auc_score(y_true, y_pred_probs)
    auc_pr = average_precision_score(y_true, y_pred_probs)

    return {
        'ACC': acc,   # Accuracy general
        'SEN': sen,   # Sensibilidad (Recall de clase positiva)
        'SPE': spe,   # Especificidad (Recall de clase negativa)
        'PPV': ppv,   # Precisión (para la clase positiva)
        'NPV': npv,   # Precisión para clase negativa
        'F1': f1,     # F1-score
        'AUC': auc,   # Área bajo curva ROC
        'AUC_PR': auc_pr  # Área bajo curva de precisión-recall
    }


def calculate_global_metrics(y_true_all, y_pred_probs_all, thresholds, fold_sizes=None):
    """
    Calcula métricas globales aplicando un umbral diferente por fold.

    Parameters
    ----------
    y_true_all : array-like
        Etiquetas verdaderas concatenadas de todos los folds.
    y_pred_probs_all : array-like
        Probabilidades predichas concatenadas.
    thresholds : list or float
        Lista de umbrales por fold o un único valor para todos.
    fold_sizes : list, optional
        Tamaño de cada fold. Si no se proporciona, se asume tamaño uniforme.

    Returns
    -------
    dict
        Métricas globales calculadas sobre todos los folds combinados.
    """
    # Si es un solo umbral para todo el dataset
    if isinstance(thresholds, float) or isinstance(thresholds, int):
        return calculate_metrics(np.array(y_true_all), np.array(y_pred_probs_all), threshold=thresholds)
    
    # Si no se especifican tamaños, asumir folds del mismo tamaño
    if fold_sizes is None:
        n = len(y_true_all)
        fold_sizes = [n // len(thresholds)] * len(thresholds)
        resto = n % len(thresholds)
        for i in range(resto):
            fold_sizes[i] += 1

    # Dividir etiquetas y probabilidades según tamaños de los folds
    indices = np.cumsum(fold_sizes)[:-1]
    y_true_folds = np.split(y_true_all, indices)
    y_probs_folds = np.split(y_pred_probs_all, indices)

    # Aplicar umbral correspondiente a cada fold
    y_pred_bin_global = np.concatenate([
        (y_probs_folds[i] >= thresholds[i]).astype(int)
        for i in range(len(thresholds))
    ])
    y_true_global = np.concatenate(y_true_folds)

    return calculate_metrics(y_true_global, y_pred_bin_global, threshold=0.5)


def save_epoch_metrics_log(logs, fold, output_dir):
    """
    Guarda en un archivo JSON el registro de métricas por época de un fold.

    Parameters
    ----------
    logs : dict
        Diccionario con métricas por época.
    fold : int
        Índice del fold actual.
    output_dir : str
        Carpeta donde guardar el archivo.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"metrics_log_fold{fold}.json")
    with open(path, 'w') as f:
        json.dump(logs, f, indent=4)


def save_validation_summary(fold_metrics, output_dir, global_metrics=None):
    """
    Guarda un resumen CSV de las métricas de validación por fold, e imprime la tabla.

    Parameters
    ----------
    fold_metrics : list of dict
        Lista con métricas de validación por fold.
    output_dir : str
        Carpeta de salida.
    global_metrics : dict, optional
        Métricas globales calculadas sobre todos los folds combinados.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Crear DataFrame con las métricas por fold
    df_metrics = pd.DataFrame(fold_metrics).round(4)
    df_metrics.index = [f"Fold {i+1}" for i in range(len(df_metrics))]

    # Agregar media y desviación estándar
    mean_row = df_metrics.mean(numeric_only=True)
    std_row = df_metrics.std(numeric_only=True)
    df_metrics.loc["Mean ± Std"] = {
        col: f"{mean_row[col]:.4f} ± {std_row[col]:.4f}" for col in df_metrics.columns
    }

    # Agregar métricas globales si existen
    if global_metrics is not None:
        global_metrics_rounded = {k: round(v, 4) for k, v in global_metrics.items()}
        df_metrics.loc["Global"] = global_metrics_rounded

    # Guardar como CSV
    df_metrics.to_csv(os.path.join(output_dir, "val_metrics_summary.csv"))

    # Mostrar por consola
    print(df_metrics)
import os
import sys
sys.path.append('../')  # Añade el directorio padre al path para importar módulos personalizados

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Importación de módulos propios del proyecto
from .models.classifier import Classifier
from utils.metrics import calculate_metrics, save_epoch_metrics_log, save_validation_summary, calculate_global_metrics
from utils.callbacks import EarlyStopping, ReduceLROnPlateau, clip_gradients
from utils.class_weights import compute_pos_weight
from utils.load_preprocess import load_data, preprocess_data
from utils.random_seed import set_seed


# Selecciona el dispositivo (GPU si está disponible, si no CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Fija la semilla para reproducibilidad
set_seed(5)


def run_classifier_training(params: dict, dataset: str, output_dir: str) -> float:
    """
    Función principal para entrenar un clasificador con validación cruzada estratificada.

    Parámetros:
    - params: diccionario con hiperparámetros y configuraciones del entrenamiento.
    - dataset: nombre o ruta del dataset a usar ('metilacion' o 'latent').
    - output_dir: directorio donde se guardarán resultados (modelos, métricas, gráficos).

    Retorna:
    - Diccionario con métricas promedio calculadas sobre todos los folds.
    """

    set_seed(5)  # Semilla para reproducibilidad al inicio

    # Crear directorios para guardar resultados si no existen
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    os.makedirs(f"{output_dir}/metrics", exist_ok=True)
    os.makedirs(f"{output_dir}/predictions", exist_ok=True)
    os.makedirs(f"{output_dir}/weights", exist_ok=True)

    # Carga los datos de entrenamiento
    X_train, y_train = load_data(dataset)

    # Inicializa validación cruzada estratificada para mantener proporción de clases en cada fold
    skf = StratifiedKFold(n_splits=params['n_folds'], shuffle=True, random_state=5)

    fold_metrics = []       # Para almacenar métricas de cada fold
    best_epoch_list = []    # Para almacenar la mejor época por fold
    all_preds = []          # Para almacenar todas las predicciones de validación
    all_targets = []        # Para almacenar todas las etiquetas verdaderas de validación

    # Itera sobre cada fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n--- Fold {fold + 1} ---")

        # Divide datos en train y validación según índices del fold
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Preprocesa y escala si no estamos trabajando con datos latentes ya procesados
        if dataset != 'latent':
            X_tr_scaled, X_val_scaled = preprocess_data(X_tr, X_val, method=params['scaling_method'])
        else:
            X_tr_scaled, X_val_scaled = X_tr, X_val

        # Crea datasets y dataloaders para PyTorch
        train_dataset = TensorDataset(torch.FloatTensor(X_tr_scaled), torch.FloatTensor(y_tr))
        val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val))

        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

        # Define el modelo clasificador
        classifier = Classifier(
            input_dim=X_tr.shape[1],
            hidden_dims=params['classifier_dims'],
            output_dim=1,
            activation=getattr(nn, params['activation_c']),
            use_batchnorm=params['use_batchnorm'],
            dropout_prob=params['dropout_prob_c']
        ).to(device)

        # Cálculo de pesos de clases para manejar desequilibrios si está habilitado
        pos_weight = compute_pos_weight(y_train, device) if params['use_class_weights'] else None

        # Definición de función de pérdida con pesos de clase si corresponde
        criterion_class = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()

        # Optimizer elegido dinámicamente según parámetro
        optimizer = getattr(optim, params['optimizer_class'])(list(classifier.parameters()), lr=params['learning_rate'])

        # Callbacks para manejo de early stopping y ajuste dinámico de learning rate
        early_stopping = EarlyStopping(
            patience=params['early_stopping_patience'],
            checkpoint_dir=f"{output_dir}/weights",
            fold=fold+1,
            min_epochs=0
        )
        lr_scheduler = ReduceLROnPlateau(optimizer, patience=params['reduce_lr_patience'])

        # Listas para logging de pérdidas y métricas por época
        epoch_logs, train_loss, val_loss = [], [], []

        # Entrenamiento por épocas
        for epoch in range(params['num_epochs']):
            classifier.train()
            train_loss_epoch = 0

            # Itera sobre batches del entrenamiento
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                optimizer.zero_grad()
                logits = classifier(batch_x).view(-1)  # Salida sin aplicar sigmoid, para BCEWithLogitsLoss
                loss = criterion_class(logits, batch_y)
                loss.backward()

                # Si está habilitado, limita el gradiente para evitar explosión
                if params['use_gradient_clipping']:
                    clip_gradients(classifier, 1.0)

                optimizer.step()
                train_loss_epoch += loss.item()

            # Validación (evaluación)
            classifier.eval()
            val_preds, val_targets = [], []
            val_loss_epoch = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    logits = classifier(batch_x).view(-1)
                    probs = torch.sigmoid(logits).cpu().numpy()  # Convierte logits a probabilidades
                    val_preds.extend(probs)
                    val_targets.extend(batch_y.numpy())
                    val_loss_epoch += criterion_class(logits, batch_y.to(device)).item()

            # Guarda promedios de pérdidas
            train_loss.append(train_loss_epoch / len(train_loader))
            val_loss.append(val_loss_epoch / len(val_loader))

            # Calcula métricas sobre la validación
            metrics = calculate_metrics(np.array(val_targets), np.array(val_preds))

            # Guarda logs por época
            epoch_logs.append({
                "epoch": epoch + 1,
                "train_loss": train_loss[-1],
                "val_loss": val_loss[-1],
                "metrics": metrics
            })

            # Ajusta learning rate si se especifica
            if params['use_reduce_lr']:
                lr_scheduler.step(val_loss[-1])

            # Early stopping para evitar overfitting y ahorrar tiempo
            if params['use_early_stopping']:
                early_stopping(val_loss[-1], {
                    "classifier": classifier
                }, epoch + 1, metrics, val_preds, val_targets)

                if early_stopping.early_stop:
                    break

        # Restaurar pesos de la mejor época según early stopping
        early_stopping.restore_best_weights({"classifier": classifier})

        # Guardar logs de métricas por época en JSON
        save_epoch_metrics_log(epoch_logs, fold=fold+1, output_dir=f"{output_dir}/metrics")

        # Matriz de confusión de la mejor época
        cm = confusion_matrix(early_stopping.best_targets, (np.array(early_stopping.best_preds) >= 0.5).astype(int))
        fig_cm, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Nevus', 'Melanoma']).plot(ax=ax)
        plt.savefig(f"{output_dir}/figures/confusion_matrix_fold{fold+1}.png")
        plt.close()

        # Curvas de pérdida (train y val) por época
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_loss, label='Train Loss')
        ax.plot(val_loss, label='Val Loss')
        ax.set_title(f'Loss Curve - Fold {fold + 1}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/figures/loss_fold{fold+1}.png")
        plt.close()

        # Guardar predicciones y etiquetas verdaderas para análisis posterior
        np.save(f"{output_dir}/predictions/val_preds_fold{fold+1}.npy", early_stopping.best_preds)
        np.save(f"{output_dir}/predictions/val_targets_fold{fold+1}.npy", early_stopping.best_targets)

        # Almacenar métricas y mejores épocas para resumen global
        fold_metrics.append(early_stopping.best_metrics)
        best_epoch_list.append(early_stopping.best_epoch)
        all_preds.extend(early_stopping.best_preds)
        all_targets.extend(early_stopping.best_targets)

    # Calcular métricas globales concatenando todas las predicciones y targets
    global_metrics = calculate_global_metrics(all_targets, all_preds, thresholds=0.5)

    # Guardar resumen global en CSV y mostrar por pantalla
    save_validation_summary(fold_metrics, output_dir=f"{output_dir}/metrics", global_metrics=global_metrics)

    # Guardar parámetros usados, incluyendo la mejor época de cada fold, en JSON
    params_to_save = params.copy()
    params_to_save["best_epoch_per_fold"] = best_epoch_list
    with open(f"{output_dir}/params.json", "w") as f:
        json.dump(params_to_save, f, indent=4)

    # Calcular y devolver métricas promedio (media simple) de todos los folds
    avg_metrics = {
        key: np.mean([fold[key] for fold in fold_metrics])
        for key in fold_metrics[0].keys()
    }

    print("Average metrics across folds:")
    for key, val in avg_metrics.items():
        print(f"{key}: {val:.4f}")

    return avg_metrics

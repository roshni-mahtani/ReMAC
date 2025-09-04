import os
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

from .models.autoencoder import DenseAutoencoder
from .models.classifier import Classifier
from utils.metrics import calculate_metrics, save_epoch_metrics_log, save_validation_summary, calculate_global_metrics
from utils.callbacks import EarlyStopping, ReduceLROnPlateau, clip_gradients
from utils.class_weights import compute_pos_weight
from utils.load_preprocess import load_data, preprocess_data
from utils.random_seed import set_seed


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(5)

def run_training(params: dict, dataset: str, output_dir: str) -> float:
    set_seed(5)

    # Crear carpetas necesarias
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    os.makedirs(f"{output_dir}/metrics", exist_ok=True)
    os.makedirs(f"{output_dir}/predictions", exist_ok=True)
    os.makedirs(f"{output_dir}/weights", exist_ok=True)

    # Cargar datos según dataset
    X_train, y_train = load_data(dataset)

    # Cross-validation
    skf = StratifiedKFold(n_splits=params['n_folds'], shuffle=True, random_state=5)

    fold_metrics = []
    best_epoch_list = []
    all_preds = []
    all_targets = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n--- Fold {fold + 1} ---")

        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        X_tr_scaled, X_val_scaled = preprocess_data(X_tr, X_val, method=params['scaling_method'])

        train_dataset = TensorDataset(torch.FloatTensor(X_tr_scaled), torch.FloatTensor(y_tr))
        val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val))

        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

        autoencoder = DenseAutoencoder(
            input_dim=X_tr.shape[1],
            encoder_dims=params['encoder_dims'],
            decoder_dims=params['encoder_dims'][:-1][::-1],
            activation=getattr(nn, params['activation_ae']),
            use_batchnorm=params['use_batchnorm'],
            dropout_prob=params['dropout_prob_ae']
        ).to(device)

        classifier = Classifier(
            input_dim=params['encoder_dims'][-1],
            hidden_dims=params['classifier_dims'],
            output_dim=1,
            activation=getattr(nn, params['activation_c']),
            use_batchnorm=params['use_batchnorm'],
            dropout_prob=params['dropout_prob_c']
        ).to(device)

        # Loss, optimizer, callbacks
        criterion_recon = nn.MSELoss()
        pos_weight = compute_pos_weight(y_train, device) if params['use_class_weights'] else None
        criterion_class = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()
        optimizer = getattr(optim, params['optimizer_class'])(list(autoencoder.parameters()) + list(classifier.parameters()), lr=params['learning_rate'])

        early_stopping = EarlyStopping(
            patience=params['early_stopping_patience'],
            checkpoint_dir=f"{output_dir}/weights",
            fold=fold+1,
            min_epochs=0
        )
        lr_scheduler = ReduceLROnPlateau(optimizer, patience=params['reduce_lr_patience'])

        # Logging
        epoch_logs, train_total_loss, val_total_loss = [], [], []
        train_recon_loss, val_recon_loss = [], []
        train_class_loss, val_class_loss = [], []

        for epoch in range(params['num_epochs']):
            autoencoder.train()
            classifier.train()
            total_loss, recon_loss_sum, class_loss_sum = 0, 0, 0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                encoded, decoded = autoencoder(batch_x)
                logits = classifier(encoded).view(-1)

                recon_loss = criterion_recon(decoded, batch_x)
                class_loss = criterion_class(logits, batch_y)
                loss = recon_loss + class_loss
                loss.backward()

                if params['use_gradient_clipping']:
                    clip_gradients(autoencoder, 1.0)
                    clip_gradients(classifier, 1.0)

                optimizer.step()

                total_loss += loss.item()
                recon_loss_sum += recon_loss.item()
                class_loss_sum += class_loss.item()

            autoencoder.eval()
            classifier.eval()
            val_preds, val_targets = [], []
            val_loss, val_recon_sum, val_class_sum = 0, 0, 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    encoded, decoded = autoencoder(batch_x)
                    logits = classifier(encoded).view(-1)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    val_preds.extend(probs)
                    val_targets.extend(batch_y.numpy())

                    val_recon = criterion_recon(decoded, batch_x)
                    val_class = criterion_class(logits, batch_y.to(device))
                    val_loss += val_recon.item() + val_class.item()
                    val_recon_sum += val_recon.item()
                    val_class_sum += val_class.item()

            train_total_loss.append(total_loss / len(train_loader))
            val_total_loss.append(val_loss / len(val_loader))
            train_recon_loss.append(recon_loss_sum / len(train_loader))
            val_recon_loss.append(val_recon_sum / len(val_loader))
            train_class_loss.append(class_loss_sum / len(train_loader))
            val_class_loss.append(val_class_sum / len(val_loader))

            metrics = calculate_metrics(np.array(val_targets), np.array(val_preds))

            epoch_logs.append({
                "epoch": epoch + 1,
                "train_loss": train_total_loss[-1],
                "val_loss": val_total_loss[-1],
                "train_recon_loss": train_recon_loss[-1],
                "val_recon_loss": val_recon_loss[-1],
                "train_class_loss": train_class_loss[-1],
                "val_class_loss": val_class_loss[-1],
                "metrics": metrics
            })

            if params['use_reduce_lr']:
                lr_scheduler.step(val_loss)

            if params['use_early_stopping']:
                early_stopping(val_loss, {
                    "autoencoder": autoencoder,
                    "classifier": classifier
                }, epoch + 1, metrics, val_preds, val_targets)
                if early_stopping.early_stop:
                    break

        early_stopping.restore_best_weights({"autoencoder": autoencoder, "classifier": classifier})

        save_epoch_metrics_log(epoch_logs, fold=fold+1, output_dir=f"{output_dir}/metrics")

        # Confusion matrix
        cm = confusion_matrix(early_stopping.best_targets, (np.array(early_stopping.best_preds) >= 0.5).astype(int))
        fig_cm, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Nevus', 'Melanoma']).plot(ax=ax)
        plt.savefig(f"{output_dir}/figures/confusion_matrix_fold{fold+1}.png")
        plt.close()

        # Plot loss curves
        fig, axs = plt.subplots(2, 2, figsize=(18, 10))

        axs[0, 0].plot(train_total_loss, label='Train Loss')
        axs[0, 0].plot(val_total_loss, label='Val Loss')
        axs[0, 0].set_title('Total Loss')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        axs[0, 1].plot(train_recon_loss, label='Train Recon Loss')
        axs[0, 1].plot(val_recon_loss, label='Val Recon Loss')
        axs[0, 1].set_title('Reconstruction Loss')
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        axs[1, 0].plot(train_class_loss, label='Train Class Loss')
        axs[1, 0].plot(val_class_loss, label='Val Class Loss')
        axs[1, 0].set_title('Classification Loss')
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        axs[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/figures/combined_losses_fold{fold+1}.png")
        plt.close()

        np.save(f"{output_dir}/predictions/val_preds_fold{fold+1}.npy", early_stopping.best_preds)
        np.save(f"{output_dir}/predictions/val_targets_fold{fold+1}.npy", early_stopping.best_targets)

        fold_metrics.append(early_stopping.best_metrics)
        best_epoch_list.append(early_stopping.best_epoch)
        all_preds.extend(early_stopping.best_preds)
        all_targets.extend(early_stopping.best_targets)

    global_metrics = calculate_global_metrics(all_targets, all_preds, thresholds=0.5)
    save_validation_summary(fold_metrics, output_dir=f"{output_dir}/metrics", global_metrics=global_metrics)

    # Guardar los parámetros
    params_to_save = params.copy()
    params_to_save["best_epoch_per_fold"] = best_epoch_list
    with open(f"{output_dir}/params.json", "w") as f:
        json.dump(params_to_save, f, indent=4)

    # Calcular y devolver métricas promedio
    avg_metrics = {
        key: np.mean([fold[key] for fold in fold_metrics])
        for key in fold_metrics[0].keys()
    }

    print("Average metrics across folds:")
    for key, val in avg_metrics.items():
        print(f"{key}: {val:.4f}")

    return avg_metrics
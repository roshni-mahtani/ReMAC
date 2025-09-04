"""
Script para encontrar el mejor espacio latente con autoencoders.

Este código realiza un grid search sobre una rejilla de hiperparámetros utilizando 
validación cruzada estratificada (StratifiedKFold con 4 folds) para autoencoders.

Para cada combinación de hiperparámetros:
  1. Entrena un modelo autoencoder por fold.
  2. Calcula el MSE medio de validación.
  3. Guarda los modelos, métricas y latentes intermedios.

Después:
  - Se selecciona la combinación con mejor MSE medio.
  - Se entrena un autoencoder final usando todos los datos y el número de épocas promedio de los folds.
  - Se guarda el espacio latente final del conjunto completo.
"""

import os
import sys
sys.path.append('../')
import shutil
import json
from itertools import product
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

from deep_learning.models.autoencoder import DenseAutoencoder
from utils.callbacks import EarlyStopping, ReduceLROnPlateau, clip_gradients
from utils.load_preprocess import load_data, preprocess_data
from utils.random_seed import set_seed

# Parámetros globales
BASE_RESULTS_DIR = "../../results/results_autoencoder"
DATASETS = ["metilacion"]

param_grid = {
    "encoder_dims": [[128, 64, 32], [64, 32, 16], [64, 32]],
    "activation": ["ReLU"],
    "dropout_prob": [0.2, 0.3],
    "use_batchnorm": [False],
    "batch_size": [4, 8],
    "num_epochs": [100],
    "learning_rate": [0.001, 0.0005],
    "optimizer_class": ["Adam"],
    "use_early_stopping": [True],
    "early_stopping_patience": [15],
    "use_reduce_lr": [True],
    "reduce_lr_patience": [10],
    "use_gradient_clipping": [True],
    "n_folds": [4],
    "scaling_method": ["divided_by_100"],
}

def generate_param_combinations(grid):
    keys, values = zip(*grid.items())
    for combination in product(*values):
        yield dict(zip(keys, combination))

def get_activation(name):
    return getattr(nn, name) if hasattr(nn, name) else nn.ReLU

def mse_loss(x, x_hat):
    return nn.MSELoss()(x_hat, x)

def train_autoencoder_fold(model, optimizer, train_loader, val_loader, device, config, fold_dir, fold_num):
    early_stopper = EarlyStopping(
        patience=config.get("early_stopping_patience", 15),
        delta=0.0,
        checkpoint_dir=fold_dir,
        fold=fold_num
    ) if config.get("use_early_stopping", True) else None

    reducer = ReduceLROnPlateau(
        optimizer,
        patience=config.get("reduce_lr_patience", 10)
    ) if config.get("use_reduce_lr", False) else None

    num_epochs = config["num_epochs"]
    best_epoch = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            _, x_hat = model(x)
            loss = mse_loss(x, x_hat)
            loss.backward()
            if config.get("use_gradient_clipping", False):
                clip_gradients(model)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                _, x_hat = model(x)
                val_loss = mse_loss(x, x_hat)
                val_losses.append(val_loss.item())

        val_loss_mean = np.mean(val_losses)
        print(f"Epoch {epoch}/{num_epochs} - Val MSE: {val_loss_mean:.6f}")

        if reducer:
            reducer.step(val_loss_mean)

        if early_stopper:
            early_stopper(val_loss_mean, {"autoencoder": model}, epoch, {}, [], [])
            if early_stopper.early_stop:
                print("Early stopping triggered")
                best_epoch = early_stopper.best_epoch
                early_stopper.restore_best_weights({"autoencoder": model})
                break
            else:
                best_epoch = epoch

    return model, best_epoch

def run_autoencoder_cv(params: dict, dataset: str, output_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(5)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/metrics", exist_ok=True)
    os.makedirs(f"{output_dir}/weights", exist_ok=True)

    X_train, y_train = load_data(dataset)

    skf = StratifiedKFold(n_splits=params['n_folds'], shuffle=True, random_state=5)
    fold_mse = []
    fold_epochs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"\n--- Fold {fold} ---")

        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        
        X_tr_scaled, X_val_scaled = preprocess_data(X_tr, X_val, method=params['scaling_method'])

        train_dataset = TensorDataset(torch.FloatTensor(X_tr_scaled))
        val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled))

        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

        autoencoder = DenseAutoencoder(
            input_dim=X_train.shape[1],
            encoder_dims=params['encoder_dims'],
            decoder_dims=params['encoder_dims'][:-1][::-1],
            activation=get_activation(params['activation']),
            use_batchnorm=params['use_batchnorm'],
            dropout_prob=params['dropout_prob']
        ).to(device)

        optimizer = getattr(torch.optim, params['optimizer_class'])(
            autoencoder.parameters(), lr=params['learning_rate'])

        fold_dir = f"{output_dir}/fold_{fold}"
        os.makedirs(fold_dir, exist_ok=True)

        autoencoder, best_epoch = train_autoencoder_fold(autoencoder, optimizer, train_loader, val_loader, device, params, fold_dir, fold)
        fold_epochs.append(best_epoch)

        # Save weights
        torch.save(autoencoder.encoder.state_dict(), f"{fold_dir}/encoder.pth")
        torch.save(autoencoder.decoder.state_dict(), f"{fold_dir}/decoder.pth")

        # Calculate MSE
        autoencoder.eval()
        with torch.no_grad():
            Z_val = autoencoder.encoder(torch.FloatTensor(X_val_scaled).to(device)).cpu().numpy()
            recon_val = autoencoder.decoder(torch.FloatTensor(Z_val).to(device)).cpu().numpy()
            mse = np.mean((recon_val - X_val_scaled) ** 2)
            fold_mse.append(mse)

            with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
                json.dump({"val_mse": mse, "best_epoch": best_epoch}, f, indent=4)

    # Save overall metrics
    avg_mse = np.mean(fold_mse)
    std_mse = np.std(fold_mse)
    avg_epochs = np.mean(fold_epochs)

    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump({
            "mse_mean": avg_mse,
            "mse_std": std_mse,
            "avg_epochs": avg_epochs
        }, f, indent=4)

    # Guardar parámetros
    with open(f"{output_dir}/params.json", "w") as f:
        json.dump(params, f, indent=4)

    print(f"\n📊 Finished CV - Mean MSE: {avg_mse:.6f} ± {std_mse:.6f}, Average Epochs: {avg_epochs:.2f}")


def train_final_model(params, dataset, output_dir, avg_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(5)

    X_train, y_train = load_data(dataset)

    X_train_scaled, _ = preprocess_data(X_train, None, method=params['scaling_method'])

    train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled))
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

    autoencoder = DenseAutoencoder(
        input_dim=X_train.shape[1],
        encoder_dims=params['encoder_dims'],
        decoder_dims=params['encoder_dims'][:-1][::-1],
        activation=get_activation(params['activation']),
        use_batchnorm=params['use_batchnorm'],
        dropout_prob=params['dropout_prob']
    ).to(device)

    optimizer = getattr(torch.optim, params['optimizer_class'])(
        autoencoder.parameters(), lr=params['learning_rate']
    )

    for epoch in range(1, int(avg_epochs) + 1):
        autoencoder.train()
        train_losses = []
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            _, x_hat = autoencoder(x)
            loss = mse_loss(x, x_hat)
            loss.backward()
            if params.get("use_gradient_clipping", False):
                clip_gradients(autoencoder)
            optimizer.step()
            train_losses.append(loss.item())
        print(f"Epoch {epoch}/{int(avg_epochs)} - Train MSE: {np.mean(train_losses):.6f}")

    os.makedirs(output_dir, exist_ok=True)
    torch.save(autoencoder.encoder.state_dict(), f"{output_dir}/encoder.pth")
    torch.save(autoencoder.decoder.state_dict(), f"{output_dir}/decoder.pth")

    autoencoder.eval()
    with torch.no_grad():
        Z_train = autoencoder.encoder(torch.FloatTensor(X_train_scaled).to(device)).cpu().numpy()

    latents_dir = os.path.join(output_dir, "latents")
    os.makedirs(latents_dir, exist_ok=True)
    df_latents = pd.DataFrame(Z_train)
    df_latents.to_csv(f"{latents_dir}/train_latents.csv", index=True)
    df_labels = pd.DataFrame(y_train, columns=["label"])
    df_labels.to_csv(f"{latents_dir}/train_labels.csv", index=True)

    train_recon = autoencoder.decoder(torch.FloatTensor(Z_train).to(device)).detach().cpu().numpy()
    train_mse = np.mean((train_recon - X_train_scaled) ** 2)

    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    with open(f"{metrics_dir}/train_metrics.json", "w") as f:
        json.dump({"train_mse": train_mse}, f, indent=4)

    print(f"✅ Full training completed.\n📊 Train MSE: {train_mse:.6f}")

def main():
    for dataset in DATASETS:
        tmp_dir = os.path.join(BASE_RESULTS_DIR, dataset, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        results = []

        print(f"\n🔎 Dataset: {dataset}")

        for i, base_params in enumerate(generate_param_combinations(param_grid), 1):
            run_name = f"run_{i:04d}"
            output_dir = os.path.join(tmp_dir, run_name)

            params = dict(base_params)

            try:
                print(f"➡️ Ejecutando {run_name}...")
                run_autoencoder_cv(params, dataset, output_dir)

                with open(os.path.join(output_dir, "summary.json"), "r") as f:
                    summary = json.load(f)

                results.append({
                    "run_name": run_name,
                    "mse_mean": summary["mse_mean"],
                    "mse_std": summary["mse_std"],
                    "avg_epochs": summary.get("avg_epochs", params["num_epochs"]),
                    "params": params,
                    "output_dir": output_dir
                })

            except Exception as e:
                print(f"❌ Error en {run_name}: {e}")
                shutil.rmtree(output_dir, ignore_errors=True)

        if not results:
            print(f"❌ No se obtuvo ningún resultado válido para {dataset}. Revisa errores previos.")
            continue

        best = sorted(results, key=lambda x: (x["mse_mean"], x["mse_std"]))[0]
        print(f"\n🏆 Mejor modelo para {dataset} : {best['run_name']} con MSE medio = {best['mse_mean']:.6f} ± {best['mse_std']:.6f}")

        final_dir = os.path.join(BASE_RESULTS_DIR, dataset, f"best_{best['run_name']}")
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        shutil.move(best["output_dir"], final_dir)

        # Entrenar modelo final con mejores hiperparámetros y promedio de épocas
        final_model_dir = os.path.join(final_dir, "full_model")
        train_final_model(best["params"], dataset, final_model_dir, best["avg_epochs"])

        # Guardar métricas del mejor modelo
        with open(os.path.join(BASE_RESULTS_DIR, dataset, "best_metrics.json"), "w") as f:
            json.dump({
                "run_name": best["run_name"],
                "mse_mean": best["mse_mean"],
                "mse_std": best["mse_std"],
                "params": best["params"]
            }, f, indent=4)

        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()

import os
import torch
import numpy as np
import json

### 1️⃣ Early Stopping ###
# Detiene el entrenamiento si la pérdida de validación no mejora después de 'patience' épocas consecutivas.
# Además, guarda los mejores pesos del modelo, métricas y predicciones asociadas.

class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, checkpoint_dir=None, fold=None, min_epochs=0):
        """
        Inicializa el mecanismo de EarlyStopping.

        Args:
            patience (int): Número de épocas a esperar sin mejora antes de detener.
            delta (float): Mínima mejora requerida para considerar que la validación ha mejorado.
            checkpoint_dir (str): Directorio donde guardar el checkpoint del modelo.
            fold (int): Número de fold, útil en validación cruzada.
            min_epochs (int): Número mínimo de épocas antes de permitir el early stopping.
        """
        self.patience = patience
        self.delta = delta
        self.checkpoint_dir = checkpoint_dir
        self.fold = fold
        self.min_epochs = min_epochs

        self.counter = 0  # Cuenta épocas sin mejora
        self.best_loss = None
        self.early_stop = False  # Flag para detener entrenamiento
        self.best_epoch = 0  # Época en que se alcanzó el mejor val_loss

        # Para guardar el mejor estado del modelo y métricas asociadas
        self.best_states = {}
        self.best_metrics = None
        self.best_preds = None
        self.best_targets = None

    def __call__(self, val_loss, models: dict, epoch: int, metrics: dict, val_preds: list, val_targets: list):
        """
        Lógica ejecutada en cada época: comprueba si hay mejora en val_loss y guarda el mejor estado.

        Args:
            val_loss (float): Pérdida de validación actual.
            models (dict): Diccionario {nombre_modelo: modelo} para guardar sus pesos.
            epoch (int): Época actual.
            metrics (dict): Métricas calculadas (ej. accuracy, AUC, etc.).
            val_preds (list): Predicciones sobre el set de validación.
            val_targets (list): Etiquetas reales del set de validación.
        """
        # Comprobamos si hay mejora en la pérdida
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0  # Reinicia el contador
            self.best_epoch = epoch
            self.best_metrics = metrics.copy()
            self.best_preds = val_preds.copy()
            self.best_targets = val_targets.copy()

            # Guardar una copia profunda del estado de cada modelo
            self.best_states = {
                name: {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                for name, model in models.items()
            }

            # Si se especifica, guardar el checkpoint en disco
            if self.checkpoint_dir is not None and self.fold is not None:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(self.checkpoint_dir, f"best_model_fold{self.fold}.pt")
                torch.save(self.best_states, checkpoint_path)

        else:
            # Si no mejora, incrementar contador
            self.counter += 1
            if self.counter >= self.patience and epoch >= self.min_epochs:
                print("Early stopping triggered.")
                self.early_stop = True

    def restore_best_weights(self, models: dict):
        """
        Restaura los mejores pesos guardados en los modelos.

        Args:
            models (dict): Diccionario {nombre_modelo: modelo} sobre los cuales aplicar los pesos guardados.
        """
        for name, model in models.items():
            if name in self.best_states:
                model.load_state_dict(self.best_states[name])


### 2️⃣ Reduce LR on Plateau ###
# Reduce la tasa de aprendizaje cuando la pérdida de validación deja de mejorar.

class ReduceLROnPlateau:
    def __init__(self, optimizer, factor=0.5, patience=5, min_lr=1e-6, delta=0.001):
        """
        Inicializa el scheduler de LR.

        Args:
            optimizer (torch.optim.Optimizer): Optimizador de PyTorch.
            factor (float): Factor por el cual se reduce el LR.
            patience (int): Épocas sin mejora antes de reducir el LR.
            min_lr (float): Valor mínimo que puede tomar el LR.
            delta (float): Mejora mínima requerida para considerar que el val_loss mejoró.
        """
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.delta = delta
        self.best_loss = None
        self.counter = 0

    def step(self, val_loss):
        """
        Llama esta función al final de cada época para actualizar el LR si es necesario.

        Args:
            val_loss (float): Pérdida de validación actual.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                for param_group in self.optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    param_group['lr'] = new_lr
                    print(f"Reducing LR: {old_lr:.6f} --> {new_lr:.6f}")
                self.counter = 0
        else:
            self.best_loss = val_loss
            self.counter = 0


### 4️⃣ Gradient Clipping ###
# Previene el problema de gradientes explosivos limitando la magnitud de los gradientes.

def clip_gradients(model, max_norm=1.0):
    """
    Aplica clipping de gradientes al modelo para evitar gradientes excesivamente grandes.

    Args:
        model (torch.nn.Module): Modelo de PyTorch.
        max_norm (float): Máxima norma permitida para los gradientes.
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

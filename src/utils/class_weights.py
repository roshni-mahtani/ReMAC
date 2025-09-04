import torch
import numpy as np

def compute_pos_weight(y_train, device):
    """
    Calcula el pos_weight para usar en funciones de pérdida como BCEWithLogitsLoss,
    útil cuando hay desbalance de clases en clasificación binaria.

    Args:
        y_train (torch.Tensor): Etiquetas binarias (0 o 1) del conjunto de entrenamiento.
        device (torch.device): Dispositivo ('cpu' o 'cuda') donde se almacenará el resultado.

    Returns:
        torch.Tensor: Valor escalar pos_weight que se pasa a BCEWithLogitsLoss.
    """
    # Contar ejemplos de clase positiva y negativa
    num_positive = (y_train == 1).sum().item()
    num_negative = (y_train == 0).sum().item()

    # Calcular peso como relación entre negativos y positivos
    pos_weight_value = num_negative / num_positive if num_positive > 0 else 1.0  # Protege contra división por cero

    # Convertir a tensor y mover al dispositivo adecuado
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32).to(device)

    return pos_weight


def compute_pos_weight_for_xgb(y_train):
    """
    Calcula scale_pos_weight para usar en XGBoost en clasificación binaria desbalanceada.

    Args:
        y_train (array-like): Etiquetas binarias (0 o 1) del conjunto de entrenamiento.

    Returns:
        float: Valor escalar scale_pos_weight que se pasa a XGBoost.
    """
    # Contar ejemplos de clase positiva y negativa
    num_positive = np.sum(y_train == 1)
    num_negative = np.sum(y_train == 0)

    # Calcular peso como relación entre negativos y positivos
    pos_weight_value = num_negative / num_positive if num_positive > 0 else 1.0

    return pos_weight_value

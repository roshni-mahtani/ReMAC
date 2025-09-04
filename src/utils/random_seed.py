import random
import numpy as np
import torch

def set_seed(seed=5):
    """
    Fija la semilla para asegurar la reproducibilidad de resultados 
    en operaciones aleatorias de Python, NumPy y PyTorch.

    Parameters
    ----------
    seed : int, optional
        Valor de la semilla aleatoria. Por defecto es 5.
    """
    # Semilla para el generador de números aleatorios de Python
    random.seed(seed)

    # Semilla para NumPy
    np.random.seed(seed)

    # Semillas para PyTorch (CPU y CUDA)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Configuraciones adicionales para garantizar determinismo en CUDA
    torch.backends.cudnn.deterministic = True  # Asegura resultados deterministas
    torch.backends.cudnn.benchmark = False     # Desactiva heurísticas de optimización que introducen aleatoriedad

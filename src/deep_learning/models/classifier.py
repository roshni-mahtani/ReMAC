import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, output_dim=1, activation=nn.ReLU, 
                 use_batchnorm=False, dropout_prob=None):
        """
        input_dim: int -> Tamaño del vector de entrada.
        hidden_dims: list[int] or None -> Lista con el número de neuronas por capa oculta. Si None, no hay capas ocultas.
        output_dim: int -> Número de clases (por defecto 1 para clasificación binaria).
        activation: torch.nn.Module -> Función de activación a usar (por defecto ReLU).
        use_batchnorm: bool -> Si se añade BatchNorm entre capas.
        dropout_prob: float or None -> Probabilidad para Dropout. None si no se quiere usar.
        """
        super(Classifier, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = []  # Sin capas ocultas
        
        layers = []
        prev_dim = input_dim
        
        # Construcción de capas ocultas
        if hidden_dims:
            for dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, dim))     # Capa lineal
                if use_batchnorm and dim > 2:               # Agregar BatchNorm solo si está activado y hay suficientes unidades
                    layers.append(nn.BatchNorm1d(dim))
                layers.append(activation())                 # Activación
                if dropout_prob is not None:                # Dropout si se especifica
                    layers.append(nn.Dropout(dropout_prob))
                prev_dim = dim
        
        # Capa de salida
        layers.append(nn.Linear(prev_dim, output_dim))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        """
        Paso hacia adelante del clasificador.

        Parámetros:
        ----------
        x : torch.Tensor
            Lote de entrada de tamaño (batch_size, input_dim)

        Returns:
        -------
        torch.Tensor
            Salida del modelo (sin aplicar sigmoid/softmax)
        """
        return self.classifier(x)

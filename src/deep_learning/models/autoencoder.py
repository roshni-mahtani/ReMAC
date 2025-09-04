import torch
import torch.nn as nn

class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim, encoder_dims, decoder_dims, activation=nn.ReLU, 
                 use_batchnorm=False, dropout_prob=None):
        """
        input_dim: int -> Número de features de entrada.
        encoder_dims: list[int] -> Lista con el número de neuronas por capa en el encoder.
        decoder_dims: list[int] -> Lista con el número de neuronas por capa en el decoder.
        activation: torch.nn.Module -> Función de activación a usar (por defecto ReLU).
        use_batchnorm: bool -> Si se añade BatchNorm entre capas.
        dropout_prob: float or None -> Probabilidad para Dropout. None si no se quiere usar.
        """
        super(DenseAutoencoder, self).__init__()
        encoder_layers = []
        decoder_layers = []

        # Encoder
        prev_dim = input_dim
        for dim in encoder_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))     # Capa lineal
            if use_batchnorm and dim > 2:                       # BatchNorm opcional
                encoder_layers.append(nn.BatchNorm1d(dim))
            encoder_layers.append(activation())                 # Activación
            if dropout_prob is not None:                        # Dropout opcional
                encoder_layers.append(nn.Dropout(dropout_prob))
            prev_dim = dim                                      # Actualizar para la siguiente capa
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        for dim in decoder_dims:
            decoder_layers.append(nn.Linear(prev_dim, dim))
            if use_batchnorm and dim > 2:
                decoder_layers.append(nn.BatchNorm1d(dim))
            decoder_layers.append(activation())
            if dropout_prob is not None:
                decoder_layers.append(nn.Dropout(dropout_prob))
            prev_dim = dim
        # Output layer con sigmoid para limitar el rango entre [0, 1]
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """
        Paso hacia adelante del autoencoder.

        Parámetros
        ----------
        x : torch.Tensor
            Lote de entrada de tamaño (batch_size, input_dim)

        Returns
        -------
        encoded : torch.Tensor
            Representación comprimida (codificada).
        decoded : torch.Tensor
            Reconstrucción de la entrada.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

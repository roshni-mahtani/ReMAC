# stdlib
from typing import Any, List, Tuple, Union

# third party
import numpy as np
import math, sys
import pandas as pd
import torch
from torch import nn
from functools import partial
import time, os, json
from utils.utils_remasker import NativeScaler, MAEDataset, adjust_learning_rate
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
import model_mae
from torch.utils.data import DataLoader, RandomSampler
import sys
import timm.optim.optim_factory as optim_factory
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.random_seed import set_seed

# hyperimpute absolute
#from hyperimpute.plugins.imputers import ImputerPlugin
#from sklearn.datasets import load_iris
#from hyperimpute.utils.benchmarks import compare_models
#from hyperimpute.plugins.imputers import Imputers

eps = 1e-8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReMasker:
    def __init__(self,
                 batch_size=8,          # Personalizar el batch_size
                 accum_iter=1,         # Iteraciones para acumulación de gradientes
                 min_lr=1e-5,          # Learning rate mínimo
                 norm_field_loss=False, # Normalización de la pérdida de campo
                 weight_decay=1e-4,    # Decaimiento de peso
                 lr=1e-3,              # Learning rate
                 blr=1e-3,             # Base learning rate
                 warmup_epochs=40,     # Épocas para el calentamiento
                 embed_dim=32,         # Dimensión de la representación latente
                 depth=4,             # Profundidad del encoder
                 decoder_depth=2,     # Profundidad del decoder
                 num_heads=4,         # Número de cabezas en el mecanismo de atención
                 mlp_ratio=4.,         # Ratio del MLP
                 max_epochs=100,       # Número de épocas para el entrenamiento
                 mask_ratio=0.5,      # Ratio de enmascarado
                 encode_func='linear'  # Función para el encoder
                 ):
        # Hiperparámetros de entrenamiento
        self.batch_size = batch_size
        self.accum_iter = accum_iter
        self.min_lr = min_lr
        self.norm_field_loss = norm_field_loss
        self.weight_decay = weight_decay
        self.lr = lr
        self.blr = blr
        self.warmup_epochs = warmup_epochs
        self.model = None
        self.norm_parameters = None

        # Arquitectura del modelo
        self.embed_dim = embed_dim
        self.depth = depth
        self.decoder_depth = decoder_depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.max_epochs = max_epochs
        self.mask_ratio = mask_ratio
        self.encode_func = encode_func

    def fit(self, X_raw: pd.DataFrame):
        """
        Entrena el modelo sobre el dataset con valores faltantes.
        """
        set_seed(5)
        X = X_raw.clone()

        # Parameters
        no = len(X)
        dim = len(X[0, :])

        X = X.cpu()

        min_val = np.zeros(dim)
        max_val = np.zeros(dim)

        for i in range(dim):
            min_val[i] = np.nanmin(X[:, i])
            max_val[i] = np.nanmax(X[:, i])
            X[:, i] = (X[:, i] - min_val[i]) / (max_val[i] - min_val[i] + eps)

        self.norm_parameters = {"min": min_val, "max": max_val}

        # Set missing
        M = 1 - (1 * (np.isnan(X)))
        M = M.float().to(device)

        X = torch.nan_to_num(X)
        X = X.to(device)

        self.model = model_mae.MaskedAutoencoder(
            rec_len=dim,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            decoder_embed_dim=self.embed_dim,
            decoder_depth=self.decoder_depth,
            decoder_num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=eps),
            norm_field_loss=self.norm_field_loss,
            encode_func=self.encode_func
        )

        # if self.improve and os.path.exists(self.path):
        #     self.model.load_state_dict(torch.load(self.path))
        #     self.model.to(device)
        #     return self

        self.model.to(device)

        # set optimizers
        # param_groups = optim_factory.add_weight_decay(model, weight_decay)
        eff_batch_size = self.batch_size * self.accum_iter
        if self.lr is None:  # only base_lr is specified
            self.lr = self.blr * eff_batch_size / 64
        # param_groups = optim_factory.add_weight_decay(self.model, self.weight_decay)
        # optimizer = torch.optim.AdamW(param_groups, lr=self.lr, betas=(0.9, 0.95))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.95))
        loss_scaler = NativeScaler()

        dataset = MAEDataset(X, M)
        dataloader = DataLoader(
            dataset, sampler=RandomSampler(dataset),
            batch_size=self.batch_size,
        )

        # if self.resume and os.path.exists(self.path):
        #     self.model.load_state_dict(torch.load(self.path))
        #     self.lr *= 0.5

        self.model.train()

        for epoch in range(self.max_epochs):

            optimizer.zero_grad()
            total_loss = 0

            iter = 0
            for iter, (samples, masks) in enumerate(dataloader):

                # we use a per iteration (instead of per epoch) lr scheduler
                if iter % self.accum_iter == 0:
                    adjust_learning_rate(optimizer, iter / len(dataloader) + epoch, self.lr, self.min_lr,
                                         self.max_epochs, self.warmup_epochs)

                samples = samples.unsqueeze(dim=1)
                samples = samples.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                # print(samples, masks)

                with torch.cuda.amp.autocast():
                    loss, _, _, _ = self.model(samples, masks, mask_ratio=self.mask_ratio)
                    loss_value = loss.item()
                    total_loss += loss_value

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)

                loss /= self.accum_iter
                loss_scaler(loss, optimizer, parameters=self.model.parameters(),
                            update_grad=(iter + 1) % self.accum_iter == 0)

                if (iter + 1) % self.accum_iter == 0:
                    optimizer.zero_grad()

            total_loss = (total_loss / (iter + 1)) ** 0.5
            # if total_loss < best_loss:
            #     best_loss = total_loss
            #     torch.save(self.model.state_dict(), self.path)
            # if (epoch + 1) % 10 == 0 or epoch == 0:
            # print((epoch+1),',', total_loss)

        # torch.save(self.model.state_dict(), self.path)
        return self

    def transform(self, X_raw: torch.Tensor):
        """
        Imputa valores faltantes en un dataset usando el modelo entrenado.
        """

        X = X_raw.clone()

        min_val = self.norm_parameters["min"]
        max_val = self.norm_parameters["max"]

        no, dim = X.shape
        X = X.cpu()

        # MinMaxScaler normalization
        for i in range(dim):
            X[:, i] = (X[:, i] - min_val[i]) / (max_val[i] - min_val[i] + eps)

        # Set missing
        M = 1 - (1 * (np.isnan(X)))
        X = np.nan_to_num(X)

        X = torch.from_numpy(X).to(device)
        M = M.to(device)

        self.model.eval()

        # Imputed data
        with torch.no_grad():
            for i in range(no):
                sample = torch.reshape(X[i], (1, 1, -1))
                mask = torch.reshape(M[i], (1, -1))
                _, pred, _, _ = self.model(sample, mask)
                pred = pred.squeeze(dim=2)
                if i == 0:
                    imputed_data = pred
                else:
                    imputed_data = torch.cat((imputed_data, pred), 0)

                    # Renormalize
        for i in range(dim):
            imputed_data[:, i] = imputed_data[:, i] * (max_val[i] - min_val[i] + eps) + min_val[i]

        if np.all(np.isnan(imputed_data.detach().cpu().numpy())):
            err = "The imputed result contains nan. This is a bug. Please report it on the issue tracker."
            raise RuntimeError(err)

        M = M.cpu()
        imputed_data = imputed_data.detach().cpu()
        # print('imputed', imputed_data, M)
        # print('imputed', M * np.nan_to_num(X_raw.cpu()) + (1 - M) * imputed_data)
        return M * np.nan_to_num(X_raw.cpu()) + (1 - M) * imputed_data

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Entrena el modelo e imputa el dataset.
        """
        X = torch.tensor(X.values, dtype=torch.float32)
        return self.fit(X).transform(X).detach().cpu().numpy()
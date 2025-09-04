from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd

def load_data(dataset):
    """
    Carga los datos de entrenamiento y test para el dataset de spitzoides.
    
    Parámetros:
    ----------
    dataset : str
        Nombre del dataset a cargar. Debe ser:
        - 'metilacion'
        - 'missing5', 'missing10', 'missing20', 'missing30', 'missing50'
        - 'latent'
    
    Returns:
    -------
    X_train : np.ndarray
        Matriz de características de entrenamiento.
    y_train : np.ndarray
        Etiquetas binarias (0: Nevus, 1: Melanoma) para entrenamiento.
    """
    if dataset == "metilacion":
        X_train = pd.read_csv("/workspace/proyecto/def_code/data/metilacion_nm.csv", index_col=0).values
    elif dataset.startswith("missing"):
        X_train = pd.read_csv(f"/workspace/proyecto/def_code/data/metilacion_with_missing/X_{dataset}_nan.csv", index_col=0).values
    elif dataset == "latent":
        X_train = pd.read_csv("/workspace/proyecto/def_code/data/latent_autoencoder.csv", index_col=0).values
    else:
        raise ValueError("El dataset debe ser 'metilacion', comenzar con 'missing', o 'latent'.")

    # Cargar etiquetas originales desde archivo CSV
    y_train_raw = pd.read_csv("/workspace/proyecto/def_code/data/labels.csv", index_col=0).iloc[:, 0].values
    # Mapear etiquetas textuales a valores binarios
    label_mapping_nm = {'Nevus': 0, 'Melanoma': 1}
    y_train = np.array([label_mapping_nm[label] for label in y_train_raw])

    return X_train, y_train


def preprocess_data(X_train, X_test, method='divided_by_100'):
    """
    Aplica un método de preprocesamiento a los datos de entrenamiento y test.

    Parámetros:
    ----------
    X_train : np.ndarray
        Matriz de características de entrenamiento.

    X_test : np.ndarray
        Matriz de características de prueba.

    method : str
        Método de preprocesamiento a aplicar. Opciones:
        - 'standard'      : Estandarización (media 0, varianza 1)
        - 'minmax'        : Escalado entre 0 y 1
        - 'divided_by_100': División directa por 100

    Returns:
    -------
    X_train_scaled : np.ndarray
        Datos de entrenamiento preprocesados.

    X_test_scaled : np.ndarray
        Datos de prueba preprocesados.
    """
    if method == 'standard':
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    elif method == 'minmax':
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    elif method == 'divided_by_100':
        X_train_scaled = X_train / 100
        X_test_scaled = (X_test / 100) if X_test is not None else None

    else:
        raise ValueError("Método no válido. Elige entre: 'standard', 'minmax', 'divided_by_100'")
    
    return X_train_scaled, X_test_scaled


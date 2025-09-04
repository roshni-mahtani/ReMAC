# CLASIFICACIÓN DE NEVUS Y MELANOMA CON MACHINE LEARNING Y DEEP LEARNING

Este proyecto tiene como objetivo desarrollar y evaluar modelos de aprendizaje automático y profundo para la clasificación de tumores melanocíticos (nevus vs. melanoma), utilizando datos de metilación. Se consideran tanto modelos clásicos de ML como arquitecturas de deep learning, y se evalúa también el impacto de los datos faltantes.

---

## ESTRUCTURA DEL PROYECTO

```text
.
├── data/
│   ├── metilacion_nm.csv                # Base de datos principal (sin faltantes)
│   ├── labels.csv                       # Etiquetas de clase (nevus / melanoma)
│   ├── latent_autoencoder.csv           # Mejor representación latente obtenida
│   └── metilacion_with_missing/         # Datos con valores faltantes generados aleatoriamente (con seed=5)
│       ├── X_missing5_nan.csv
│       ├── X_missing10_nan.csv
│       ├── X_missing20_nan.csv
│       ├── X_missing30_nan.csv
│       └── X_missing50_nan.csv
│
├── src/
│   ├── deep_learning/                   # Modelos y utilidades de deep learning
│   ├── machine_learning/                # Modelos y utilidades de ML clásico
│   ├── train/
│   │   ├── gridsearch.py                # Ejecuta búsquedas de hiperparámetros
│   │   └── train_optimal_models.py      # Entrena modelos con los mejores hiperparámetros
│   ├── utils/                           # Funciones auxiliares y utilidades
│   │   ├── callback.py
│   │   ├── class_weights.py
│   │   ├── metrics.py
│   │   ├── random_seed.py
│   │   ├── load_preprocess.py
│   │   ├── visualization_utils.py
│   │   ├── remasker_utils.py
│   │   ├── tablas_remasker.py
│   │   ├── create_missing.ipynb         # Generación de bases con datos faltantes
│   │   └── obtain_best_latent.py        # Obtención de mejor representación latente
│   └── requirements.txt                 # Dependencias del proyecto
│
├── results/                             # Resultados del gridsearch
├── optim_results/                       # Resultados de los modelos óptimos

```

---

## INSTALACIÓN Y USO

### 1. Entorno de ejecución

El proyecto fue desarrollado y probado en el contenedor oficial de NVIDIA:

- Imagen Docker: nvcr.io/nvidia/pytorch:23.10-py3

Puedes levantar el entorno con Docker, montar la carpeta del proyecto y acceder al contenedor para ejecutar los scripts.

Instala las dependencias necesarias con:

    pip install -r requirements.txt

---

### 2. GridSearch de modelos

Para ejecutar la búsqueda de hiperparámetros:

    python gridsearch.py <modo>

Modos disponibles:
- autoencoder_classifier
- classifier
- remasker_classifier
- ml_models (modelos clásicos sin datos faltantes)
- ml_missing (modelos compatibles con datos faltantes)

Ejemplo:

    python gridsearch.py remasker_classifier

---

### 3. Entrenamiento de los mejores modelos

Ejecuta el script correspondiente:

    python train_optimal_models.py -m <modo>

Ejemplo:

    python train_optimal_models.py -m classifier

Si no se incluye el modo, se ejecutan todos los modelos.
Los scripts se ejecutan desde la carpeta en la que se encuentran.

---

## MODELOS UTILIZADOS

### Deep Learning
- Autoencoder + Clasificador
- Clasificador
- Remasker + Clasificador (con datos faltantes)

### Machine Learning

#### Sin datos faltantes:
- Regresión Logística
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

#### Con datos faltantes:
- CatBoost
- XGBoost
- HistGradientBoostingTrees
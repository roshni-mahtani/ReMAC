# Masked Autoencoder Joint Learning for Robust Spitzoid Tumor Classification

### TL;DR
Accurate diagnosis of spitzoid tumors (ST) is critical to ensure a favorable prognosis and to avoid both under- and over-treatment. Epigenetic data, particularly DNA methylation, provide a valuable source of information for this task. However, prior studies assume complete data, an unrealistic setting as methylation profiles frequently contain missing entries due to limited coverage and experimental artifacts. Our work challenges these favorable scenarios and introduces ReMAC, an extension of ReMasker designed to tackle classification tasks on high-dimensional data under complete and incomplete regimes. Evaluation on real clinical data demonstrates that ReMAC achieves strong and robust performance compared to competing classification methods in the stratification of ST.

<p align="center">
  <img src="figures/main_figure.png" alt="Main figure" width="1000"/>
</p>

*<a href="https://scholar.google.com/citations?user=N8Y3mGAAAAAJ&hl=es" style="color:blue;">Ilán Carretero</a>, 
<a href="https://www.linkedin.com/in/roshni-mahtani-vashdev-165aa7225/?locale=en_US&trk=people-guest_people_search-card" style="color:blue;">Roshni Mahtani</a>, 
<a href="https://www.incliva.es/" style="color:blue;">Silvia Perez-Deben</a>, 
<a href="https://www.incliva.es/" style="color:blue;">José Francisco González-Muñoz</a>, 
<a href="https://www.incliva.es/" style="color:blue;">Carlos Monteagudo</a>, 
<a href="https://scholar.google.com/citations?user=jk4XsG0AAAAJ&hl=es" style="color:blue;">Valery Naranjo</a>, 
<a href="https://scholar.google.com/citations?user=CPCZPNkAAAAJ&hl=es" style="color:blue;">Rocío del Amor</a>*

📜 <span style="color:red"><em>Accepted in <a href="https://caseib.es/2025/" style="color:red;">CASEIB'25</a></em></span> 

🔗 <span><em><a href="[https://arxiv.org/pdf/2412.04260](https://arxiv.org/pdf/2511.19535)" style="color:orange;">Masked Autoencoder Joint Learning for Robust Spitzoid Tumor
Classification</a></em></span> 

---

## PROJECT STRUCTURE

```text
.
├── data/
│   ├── metilacion_nm.csv                # Main database (no missing values)
│   ├── labels.csv                       # Class labels (nevus / melanoma)
│   ├── latent_autoencoder.csv           # Best latent representation obtained
│   └── metilacion_with_missing/         # Data with randomly generated missing values (seed=5)
│       ├── X_missing5_nan.csv
│       ├── X_missing10_nan.csv
│       ├── X_missing20_nan.csv
│       ├── X_missing30_nan.csv
│       └── X_missing50_nan.csv
│
├── src/
│   ├── deep_learning/                   # Deep learning models and utilities
│   ├── machine_learning/                # Classical ML models and utilities
│   ├── train/
│   │   ├── gridsearch.py                # Runs hyperparameter searches
│   │   └── train_optimal_models.py      # Trains models with the best hyperparameters
│   ├── utils/                           # Auxiliary functions and utilities
│   │   ├── callback.py
│   │   ├── class_weights.py
│   │   ├── metrics.py
│   │   ├── random_seed.py
│   │   ├── load_preprocess.py
│   │   ├── visualization_utils.py
│   │   ├── remasker_utils.py
│   │   ├── tablas_remasker.py
│   │   ├── create_missing.ipynb         # Generation of datasets with missing data
│   │   └── obtain_best_latent.py        # Extraction of the best latent representation
│   └── requirements.txt                 # Project dependencies
│
├── results/                             # Grid search results
├── optim_results/                       # Optimal model results

```

---

## INSTALLATION AND USAGE

### 1. Runtime Environment

The project was developed and tested in the official NVIDIA container:

- Docker Image: nvcr.io/nvidia/pytorch:23.10-py3

You can launch the environment with Docker, mount the project folder, and access the container to run the scripts.

Install the required dependencies with:

    pip install -r requirements.txt

---

### 2. Model GridSearch

To run the hyperparameter search:

    python gridsearch.py <modo>

Available modes:
- autoencoder_classifier  
- classifier  
- remasker_classifier  
- ml_models (classical models without missing data)  
- ml_missing (models compatible with missing data)  

Example:

    python gridsearch.py remasker_classifier

---

### 3. Training the Best Models

Run the corresponding script:

    python train_optimal_models.py -m <modo>

Example:

    python train_optimal_models.py -m classifier

If no mode is specified, all models are executed.  
Scripts must be run from the folder where they are located.

---

## MODELS USED

### Deep Learning
- Autoencoder + Classifier  
- Classifier  
- ReMasker + Classifier (with missing data)  

### Machine Learning

#### Without missing data:
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Random Forest  
- XGBoost  

#### With missing data:
- CatBoost  
- XGBoost  
- HistGradientBoostingTrees  

---

The authors sincerely thank the researchers of the *ReMasker* method for sharing their code, available at: [https://github.com/tydusky/remasker](https://github.com/alps-lab/remasker)

---

## CITATION

If you find this code or our methodology useful in your research, please consider citing our work:

```bibtex
@article{carretero2025masked,
  title={Masked Autoencoder Joint Learning for Robust Spitzoid Tumor Classification},
  author={Carretero, Il{\'a}n and Mahtani, Roshni and Perez-Deben, Silvia and Gonz{\'a}lez-Mu{\~n}oz, Jos{\'e} Francisco and Monteagudo, Carlos and Naranjo, Valery and del Amor, Roc{\'\i}o},
  journal={arXiv preprint arXiv:2511.19535},
  year={2025}
}
```


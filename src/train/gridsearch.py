import sys
import os
import shutil
import json
from itertools import product

def generate_param_combinations(grid):
    keys, values = zip(*grid.items())
    for combination in product(*values):
        yield dict(zip(keys, combination))

def save_model(model, folder_path):
    shutil.copytree(model["output_dir"], folder_path)
    with open(os.path.join(folder_path, "metrics.json"), "w") as f:
        json.dump({
            "run_name": model["run_name"],
            "metrics": model["metrics"],
            "params": model["params"]
        }, f, indent=4)

def gridsearch_autoencoder_classifier():
    """
    Ejecuta un grid search para el modelo autoencoder + clasificador.
    Para cada combinación de parámetros entrena el modelo,
    guarda resultados y finalmente guarda el mejor modelo.
    """
    sys.path.append('../')
    from deep_learning.train_autoencoder_classifier import run_training

    BASE_RESULTS_DIR = "../../results/results_autoencoder_classifier"
    DATASETS = ["metilacion"]

    param_grid = {
        "encoder_dims": [[64, 32], [64, 32, 16], [128, 64, 32]],
        "classifier_dims": [[], [8], [16]],
        "activation_ae": ["ReLU"],
        "activation_c": ["ReLU"],
        "dropout_prob_ae": [0.2, 0.3],
        "dropout_prob_c": [0.2, 0.3],
        "use_batchnorm": [False],
        "batch_size": [4, 8],
        "num_epochs": [100],
        "learning_rate": [0.001, 0.0005],
        "optimizer_class": ["Adam"],
        "use_class_weights": [True],
        "use_early_stopping": [True],
        "use_reduce_lr": [True, False],
        "use_gradient_clipping": [True],
        "early_stopping_patience": [15],
        "reduce_lr_patience": [10],
        "n_folds": [4],
        "scaling_method": ["divided_by_100"]
    }

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
                metrics = run_training(params, dataset, output_dir)
                results.append({
                    "run_name": run_name,
                    "metrics": metrics,
                    "params": params,
                    "output_dir": output_dir
                })
            except Exception as e:
                print(f"❌ Error en {run_name}: {e}")
                shutil.rmtree(output_dir, ignore_errors=True)

        if not results:
            print("No se obtuvieron resultados válidos.")
            return

        results_sorted = sorted(results, key=lambda x: (
            -x["metrics"].get("F1", 0),
            -x["metrics"].get("AUC", 0),
            -x["metrics"].get("ACC", 0)
        ))

        best_model = results_sorted[0]
        best_dir = os.path.join(BASE_RESULTS_DIR, dataset, "best_run")

        if os.path.exists(best_dir):
            shutil.rmtree(best_dir)

        save_model(best_model, best_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"Mejor modelo guardado en: {best_dir}")

def gridsearch_classifier():
    """
    Similar a run_autoencoder_classifier, pero para modelos de solo clasificador.
    Ejecuta grid search, guarda resultados y copia el mejor modelo.
    """
    sys.path.append('../')
    from deep_learning.train_classifier import run_classifier_training

    BASE_RESULTS_DIR = "../../results/results_classifier"
    DATASETS = ["metilacion", "latent"]

    param_grid = {
        "classifier_dims": [[], [8], [16], [32], [16, 8], [32, 16]],
        "activation_c": ["ReLU"],
        "dropout_prob_c": [0.2, 0.3],
        "use_batchnorm": [False],
        "batch_size": [4, 8],
        "num_epochs": [100],
        "learning_rate": [0.001, 0.0005],
        "optimizer_class": ["Adam"],
        "use_class_weights": [True],
        "use_early_stopping": [True],
        "use_reduce_lr": [True, False],
        "use_gradient_clipping": [True],
        "early_stopping_patience": [15],
        "reduce_lr_patience": [10],
        "n_folds": [4],
        "scaling_method": ["divided_by_100"]
    }

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
                metrics = run_classifier_training(params, dataset, output_dir)
                results.append({
                    "run_name": run_name,
                    "metrics": metrics,
                    "params": params,
                    "output_dir": output_dir
                })
            except Exception as e:
                print(f"❌ Error en {run_name}: {e}")
                shutil.rmtree(output_dir, ignore_errors=True)

        if not results:
            print("⚠️ No hay resultados válidos para este grupo.")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            continue

        results_sorted = sorted(results, key=lambda x: (
            -x["metrics"].get("F1", 0),
            -x["metrics"].get("AUC", 0),
            -x["metrics"].get("ACC", 0)
        ))

        best_model = results_sorted[0]
        final_dir = os.path.join(BASE_RESULTS_DIR, dataset, "best_run")

        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        save_model(best_model, final_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"Mejor modelo guardado en: {final_dir}")

def gridsearch_remasker_classifier():
    """
    Grid search para remasker + clasificador, que lidia con datos faltantes,
    con varias combinaciones de dataset, fuente y estrategia de embeddings y ratio de máscara.
    Guarda el mejor modelo para cada configuración.
    """
    sys.path.append('../')
    from deep_learning.train_remasker_classifier import run_training

    BASE_RESULTS_DIR = "../../results/results_remasker_classifier_gs"
    DATASETS = ["metilacion"] # "metilacion", "missing5", "missing10", "missing20", "missing30", "missing50"
    EMBEDDING_SOURCES = ["decoder"] # "encoder", "decoder"
    EMBEDDING_STRATEGIES = ["cls", "mean_wo_cls"]
    MASK_RATIO = [0.5] # 0.1, 0.3, 0.5, 0.7

    param_grid = {
        'n_folds': [4],
        'scaling_method': ["divided_by_100"],
        'embed_dim': [32],
        'depth': [4],
        'decoder_depth': [2],
        'num_heads': [4],
        'mlp_ratio': [4.],
        "classifier_dims": [[], [8], [16], [32]],
        "activation_c": ["ReLU"],
        "dropout_prob_c": [0.0, 0.1, 0.2, 0.3],
        "use_batchnorm": [False],
        "batch_size": [4, 8],
        "num_epochs": [100],
        "learning_rate": [0.0005, 0.001],
        "optimizer_class": ["Adam"],
        "use_class_weights": [True],
        "use_early_stopping": [True],
        "use_reduce_lr": [False, True],
        "use_gradient_clipping": [True],
        "early_stopping_patience": [15],
        "reduce_lr_patience": [10],
        "min_epochs": [20]
    }

    for dataset in DATASETS:
        for embedding_source in EMBEDDING_SOURCES:
            for embedding_strategy in EMBEDDING_STRATEGIES:
                for mask_ratio in MASK_RATIO:
                    mask_ratio_dir = f"mask{int(mask_ratio * 100)}"
                    tmp_dir = os.path.join(BASE_RESULTS_DIR, dataset, embedding_source, embedding_strategy, mask_ratio_dir, "tmp")
                    os.makedirs(tmp_dir, exist_ok=True)

                    results = []
                    print(f"\n🔎 Dataset: {dataset} | Embedding Source: {embedding_source} | Embedding Strategy: {embedding_strategy} | Mask Ratio: {mask_ratio}")

                    for i, base_params in enumerate(generate_param_combinations(param_grid), 1):
                        run_name = f"run_{i:04d}"
                        output_dir = os.path.join(tmp_dir, run_name)
                        params = dict(base_params)
                        params.update({
                            "embedding_source": embedding_source,
                            "embedding_strategy": embedding_strategy,
                            "mask_ratio": mask_ratio
                        })

                        try:
                            print(f"➡️ Ejecutando {run_name}...")
                            metrics = run_training(params, dataset, output_dir)
                            results.append({
                                "run_name": run_name,
                                "metrics": metrics,
                                "params": params,
                                "output_dir": output_dir
                            })
                        except Exception as e:
                            print(f"❌ Error en {run_name}: {e}")
                            shutil.rmtree(output_dir, ignore_errors=True)

                    if not results:
                        print("⚠️ No hay resultados válidos para este grupo.")
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                        continue

                    results_sorted = sorted(results, key=lambda x: (
                        -x["metrics"].get("F1", 0),
                        -x["metrics"].get("AUC", 0),
                        -x["metrics"].get("ACC", 0)
                    ))

                    best_model = results_sorted[0]
                    final_dir = os.path.join(BASE_RESULTS_DIR, dataset, embedding_source, embedding_strategy, mask_ratio_dir, "best_run")

                    if os.path.exists(final_dir):
                        shutil.rmtree(final_dir)
                    save_model(best_model, final_dir)
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    print(f"Mejor modelo guardado en: {final_dir}")

def gridsearch_ml_models():
    """
    Gridsearch para modelos de machine learning:
    Regresión Logística, kNN, SVM, Random Forest y XGBoost.
    Ejecuta grid search, guarda resultados y copia el mejor modelo.
    """
    sys.path.append('../')
    from machine_learning.train_ml_models import run_ml_training

    BASE_RESULTS_DIR = "../../results/results_ml"
    DATASETS = ["metilacion", "latent"]
    FOLDS = [4]
    SCALING_METHODS = ["divided_by_100"]

    param_grid = {
        "logistic": {"C": [0.01, 0.1, 1.0, 10.0]},
        "knn": {"n_neighbors": [3, 5]},
        "svm": {"C": [0.1, 1.0, 10.0], "kernel": ["linear", "rbf"]},
        "random_forest": {"n_estimators": [50, 100], "max_depth": [3, 5, None]},
        "xgboost": {"n_estimators": [50, 100], "max_depth": [3, 5, None], "learning_rate": [0.01, 0.05, 0.1]}
    }

    for dataset in DATASETS:
        for n_folds in FOLDS:
            fold_dir = f"{n_folds}-fold"
            for scaling in SCALING_METHODS:
                for model_name, model_params in param_grid.items():
                    tmp_dir = os.path.join(BASE_RESULTS_DIR, dataset, model_name, "tmp")
                    os.makedirs(tmp_dir, exist_ok=True)
                    results = []

                    print(f"\n🔎 Dataset: {dataset} | {fold_dir} | Scaling: {scaling} | Model: {model_name}")

                    for i, combo in enumerate(generate_param_combinations(model_params), 1):
                        run_name = f"run_{i:04d}"
                        output_dir = os.path.join(tmp_dir, run_name)
                        os.makedirs(output_dir, exist_ok=True)
                        try:
                            print(f"➡️ Ejecutando {run_name}")
                            metrics = run_ml_training(
                                model_name=model_name,
                                model_params=combo,
                                dataset=dataset,
                                n_folds=n_folds,
                                scaling_method=scaling,
                                output_dir=output_dir
                            )
                            results.append({
                                "run_name": run_name,
                                "metrics": metrics,
                                "params": combo,
                                "output_dir": output_dir
                            })
                        except Exception as e:
                            print(f"❌ Error en {run_name}: {e}")
                            shutil.rmtree(output_dir, ignore_errors=True)

                    if not results:
                        print("⚠️ No hay resultados válidos para este modelo.")
                        continue

                    best_model = sorted(results, key=lambda x: (-x["metrics"]["F1"], -x["metrics"]["AUC"], -x["metrics"]["ACC"]))[0]
                    final_dir = os.path.join(BASE_RESULTS_DIR, dataset, model_name, "best_model")
                    save_model(best_model, final_dir)
                    shutil.rmtree(tmp_dir, ignore_errors=True)

def gridsearch_ml_missing():
    """
    Gridsearch para modelos de machine learning que lidian con valores faltantes:
    CatBoost, XGBoost y HistGradientBoosting.
    Ejecuta grid search, guarda resultados y copia el mejor modelo.
    """
    sys.path.append('../')
    from machine_learning.train_ml_missing import run_ml_training

    BASE_RESULTS_DIR = "../../results/results_ml_missing"
    DATASETS = ["metilacion", "missing5", "missing10", "missing20", "missing30", "missing50"]
    FOLDS = [4]
    SCALING_METHODS = ["divided_by_100"]

    param_grid = {
        "catboost": {"iterations": [50, 100], "depth": [3, 5, None], "learning_rate": [0.01, 0.05, 0.1]},
        "xgboost": {"n_estimators": [50, 100], "max_depth": [3, 5, None], "learning_rate": [0.01, 0.05, 0.1]},
        "histgb": {"max_iter": [50, 100], "max_depth": [3, 5, None], "min_samples_leaf": [2, 5], "learning_rate": [0.01, 0.05, 0.1]}
    }

    for dataset in DATASETS:
        for n_folds in FOLDS:
            fold_dir = f"{n_folds}-fold"
            for scaling in SCALING_METHODS:
                for model_name, model_params in param_grid.items():
                    tmp_dir = os.path.join(BASE_RESULTS_DIR, dataset, model_name, "tmp")
                    os.makedirs(tmp_dir, exist_ok=True)
                    results = []

                    print(f"\n🔎 Dataset: {dataset} | {fold_dir} | Scaling: {scaling} | Model: {model_name}")

                    for i, combo in enumerate(generate_param_combinations(model_params), 1):
                        run_name = f"run_{i:04d}"
                        output_dir = os.path.join(tmp_dir, run_name)
                        os.makedirs(output_dir, exist_ok=True)
                        print(f"\n➡️ {dataset} | {model_name} | {run_name}")

                        try:
                            metrics = run_ml_training(
                                model_name=model_name,
                                model_params=combo,
                                dataset=dataset,
                                n_folds=n_folds,
                                scaling_method=scaling,
                                output_dir=output_dir
                            )
                            results.append({
                                "run_name": run_name,
                                "metrics": metrics,
                                "params": combo,
                                "output_dir": output_dir
                            })
                        except Exception as e:
                            print(f"❌ Error en {run_name}: {e}")
                            shutil.rmtree(output_dir, ignore_errors=True)

                    if not results:
                        print(f"No se obtuvieron resultados para {dataset} {n_folds}-fold {scaling} {model_name}")
                        continue

                    results_nonzero = [r for r in results if all(v > 0 for v in r["metrics"].values())]
                    if results_nonzero:
                        best_model = sorted(results_nonzero, key=lambda x: (-x["metrics"]["F1"], -x["metrics"]["AUC"], -x["metrics"]["ACC"]))[0]
                        print("⚠️ Se encontró al menos un modelo con métricas > 0")
                    else:
                        best_model = sorted(results, key=lambda x: (-x["metrics"]["F1"], -x["metrics"]["AUC"], -x["metrics"]["ACC"]))[0]
                        print("⚠️ No hay modelos con métricas > 0, devolviendo el mejor de todos")

                    final_dir = os.path.join(BASE_RESULTS_DIR, dataset, model_name, "best_model")
                    save_model(best_model, final_dir)
                    shutil.rmtree(tmp_dir, ignore_errors=True)

def main(mode):
    if mode == "autoencoder_classifier":
        gridsearch_autoencoder_classifier()
    elif mode == "classifier":
        gridsearch_classifier()
    elif mode == "remasker_classifier":
        gridsearch_remasker_classifier()
    elif mode == "ml_models":
        gridsearch_ml_models()
    elif mode == "ml_missing":
        gridsearch_ml_missing()
    else:
        raise ValueError(f"Modo desconocido: {mode}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ejecutar gridsearch según tipo de modelo")
    parser.add_argument("mode", choices=["autoencoder_classifier", "classifier", "remasker_classifier", "ml_models", "ml_missing"], help="Modo de entrenamiento")
    args = parser.parse_args()

    main(args.mode)

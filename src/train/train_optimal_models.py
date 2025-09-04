import sys
import os
import shutil
import json

def save_model(model, folder_path):
    shutil.copytree(model["output_dir"], folder_path)
    with open(os.path.join(folder_path, "metrics.json"), "w") as f:
        json.dump({
            "run_name": model["run_name"],
            "metrics": model["metrics"],
            "params": model["params"]
        }, f, indent=4)

def train_autoencoder_classifier():
    sys.path.append('../')
    from deep_learning.train_autoencoder_classifier import run_training

    BASE_RESULTS_DIR = "../../optim_results/results_autoencoder_classifier"
    DATASETS = ["metilacion"]

    best_params_autoencoder_classifier = {
        "metilacion": {
            "encoder_dims": [64, 32, 16],
            "classifier_dims": [8],
            "activation_ae": "ReLU",
            "activation_c": "ReLU",
            "dropout_prob_ae": 0.3,
            "dropout_prob_c": 0.2,
            "use_batchnorm": False,
            "batch_size": 8,
            "num_epochs": 100,
            "learning_rate": 0.001,
            "optimizer_class": "Adam",
            "use_class_weights": True,
            "use_early_stopping": True,
            "use_reduce_lr": False,
            "use_gradient_clipping": True,
            "early_stopping_patience": 15,
            "reduce_lr_patience": 10,
            "n_folds": 4,
            "scaling_method": "divided_by_100"
        }
    }

    for dataset in DATASETS:
        tmp_dir = os.path.join(BASE_RESULTS_DIR, dataset, "tmp")
        best_dir = os.path.join(BASE_RESULTS_DIR, dataset, "best_run")
        os.makedirs(tmp_dir, exist_ok=True)

        run_name = "best_run_manual"
        output_dir = os.path.join(tmp_dir, run_name)
        params = best_params_autoencoder_classifier[dataset]

        try:
            print(f"🚀 Ejecutando entrenamiento óptimo para {dataset}...")
            metrics = run_training(params, dataset, output_dir)

            best_model = {
                "run_name": run_name,
                "metrics": metrics,
                "params": params,
                "output_dir": output_dir
            }

            if os.path.exists(best_dir):
                shutil.rmtree(best_dir)
            save_model(best_model, best_dir)
            print(f"✅ Modelo óptimo guardado en: {best_dir}")

        except Exception as e:
            print(f"❌ Error entrenando modelo óptimo: {e}")
            shutil.rmtree(output_dir, ignore_errors=True)

        # Limpieza de tmp
        shutil.rmtree(tmp_dir, ignore_errors=True)


def train_classifier():
    sys.path.append('../')
    from deep_learning.train_classifier import run_classifier_training

    BASE_RESULTS_DIR = "../../optim_results/results_classifier"
    DATASETS = ["metilacion", "latent"]

    best_params_classifier = {
        "metilacion": {
            "classifier_dims": [32],
            "activation_c": "ReLU",
            "dropout_prob_c": 0.2,
            "use_batchnorm": False,
            "batch_size": 8,
            "num_epochs": 100,
            "learning_rate": 0.001,
            "optimizer_class": "Adam",
            "use_class_weights": True,
            "use_early_stopping": True,
            "use_reduce_lr": True,
            "use_gradient_clipping": True,
            "early_stopping_patience": 15,
            "reduce_lr_patience": 10,
            "n_folds": 4,
            "scaling_method": "divided_by_100"
        },
        "latent": {
            "classifier_dims": [32],
            "activation_c": "ReLU",
            "dropout_prob_c": 0.3,
            "use_batchnorm": False,
            "batch_size": 8,
            "num_epochs": 100,
            "learning_rate": 0.001,
            "optimizer_class": "Adam",
            "use_class_weights": True,
            "use_early_stopping": True,
            "use_reduce_lr": True,
            "use_gradient_clipping": True,
            "early_stopping_patience": 15,
            "reduce_lr_patience": 10,
            "n_folds": 4,
            "scaling_method": "divided_by_100"
        }
    }

    for dataset in DATASETS:
        print(f"\n🚀 Ejecutando modelo óptimo para: {dataset}")
        tmp_dir = os.path.join(BASE_RESULTS_DIR, dataset, "tmp")
        final_dir = os.path.join(BASE_RESULTS_DIR, dataset, "best_run")
        os.makedirs(tmp_dir, exist_ok=True)

        run_name = "best_run_manual"
        output_dir = os.path.join(tmp_dir, run_name)
        params = best_params_classifier[dataset]

        try:
            metrics = run_classifier_training(params, dataset, output_dir)

            best_model = {
                "run_name": run_name,
                "metrics": metrics,
                "params": params,
                "output_dir": output_dir
            }

            if os.path.exists(final_dir):
                shutil.rmtree(final_dir)
            save_model(best_model, final_dir)
            print(f"✅ Modelo óptimo guardado en: {final_dir}")

        except Exception as e:
            print(f"❌ Error entrenando {dataset}: {e}")
            shutil.rmtree(output_dir, ignore_errors=True)

        shutil.rmtree(tmp_dir, ignore_errors=True)

def train_remasker_classifier():
    sys.path.append('../')
    from deep_learning.train_remasker_classifier import run_training

    BASE_RESULTS_DIR = "../../optim_results/results_remasker_classifier"
    DATASETS = ["metilacion", "missing5", "missing10", "missing20", "missing30", "missing50"]
    EMBEDDING_SOURCES = ["encoder", "decoder"]
    EMBEDDING_STRATEGIES = ["cls", "mean_wo_cls", "max_wo_cls"]
    MASK_RATIO = [0.1, 0.3, 0.5, 0.7]

    # Mejor combinación que decidiste quedarte (ajustala si hace falta)
    best_fixed_params = {
        "n_folds": 4,
        "scaling_method": "divided_by_100",
        "embed_dim": 32,
        "depth": 4,
        "decoder_depth": 2,
        "num_heads": 4,
        "mlp_ratio": 4.,
        "classifier_dims": [],
        "activation_c": "ReLU",
        "dropout_prob_c": 0.2,
        "use_batchnorm": False,
        "batch_size": 4,
        "num_epochs": 100,
        "learning_rate": 0.0005,
        "optimizer_class": "Adam",
        "use_class_weights": True,
        "use_early_stopping": True,
        "use_reduce_lr": False,
        "use_gradient_clipping": True,
        "early_stopping_patience": 50,
        "reduce_lr_patience": 25,
        "min_epochs": 20
    }

    for dataset in DATASETS:
        for embedding_source in EMBEDDING_SOURCES:
            for embedding_strategy in EMBEDDING_STRATEGIES:
                for mask_ratio in MASK_RATIO:
                    print(f"\n🚀 Ejecutando modelo óptimo para: {dataset} | {embedding_source} | {embedding_strategy} | mask_ratio={mask_ratio}")

                    mask_ratio_dir = f"mask{int(mask_ratio * 100)}"
                    tmp_dir = os.path.join(BASE_RESULTS_DIR, dataset, embedding_source, embedding_strategy, mask_ratio_dir, "tmp")
                    final_dir = os.path.join(BASE_RESULTS_DIR, dataset, embedding_source, embedding_strategy, mask_ratio_dir, "best_run")

                    os.makedirs(tmp_dir, exist_ok=True)

                    params = dict(best_fixed_params)  # Copiar parámetros fijos
                    params.update({
                        "embedding_source": embedding_source,
                        "embedding_strategy": embedding_strategy,
                        "mask_ratio": mask_ratio
                    })

                    run_name = "best_run_manual"
                    output_dir = os.path.join(tmp_dir, run_name)

                    try:
                        metrics = run_training(params, dataset, output_dir)

                        best_model = {
                            "run_name": run_name,
                            "metrics": metrics,
                            "params": params,
                            "output_dir": output_dir
                        }

                        if os.path.exists(final_dir):
                            shutil.rmtree(final_dir)
                        save_model(best_model, final_dir)
                        print(f"✅ Modelo óptimo guardado en: {final_dir}")

                    except Exception as e:
                        print(f"❌ Error entrenando {dataset} | {embedding_source} | {embedding_strategy} | mask_ratio={mask_ratio}: {e}")
                        shutil.rmtree(output_dir, ignore_errors=True)

                    shutil.rmtree(tmp_dir, ignore_errors=True)

def train_ml_models():
    sys.path.append('../')
    from machine_learning.train_ml_models import run_ml_training

    BASE_RESULTS_DIR = "../../optim_results/results_ml"
    DATASETS = ["metilacion", "latent"]

    best_params_ml = {
        "metilacion": {
            "logistic": {"C": 10.0},
            "knn": {"n_neighbors": 5},
            "svm": {"C": 0.1, "kernel": "linear"},
            "random_forest": {"n_estimators": 100, "max_depth": 5},
            "xgboost": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05}
        },
        "latent": {
            "logistic": {"C": 10.0},
            "knn": {"n_neighbors": 5},
            "svm": {"C": 0.1, "kernel": "linear"},
            "random_forest": {"n_estimators": 100, "max_depth": 3},
            "xgboost": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05}
        }
    }

    FOLDS = 4
    SCALING_METHOD = "divided_by_100"

    for dataset in DATASETS:
        for model_name, params in best_params_ml[dataset].items():
            print(f"\n🚀 Entrenando modelo óptimo: Dataset={dataset}, Modelo={model_name}")
            tmp_dir = os.path.join(BASE_RESULTS_DIR, dataset, model_name, "tmp")
            final_dir = os.path.join(BASE_RESULTS_DIR, dataset, model_name, "best_model")
            os.makedirs(tmp_dir, exist_ok=True)
            
            run_name = "best_run_manual"
            output_dir = os.path.join(tmp_dir, run_name)
            os.makedirs(output_dir, exist_ok=True)

            try:
                metrics = run_ml_training(
                    model_name=model_name,
                    model_params=params,
                    dataset=dataset,
                    n_folds=FOLDS,
                    scaling_method=SCALING_METHOD,
                    output_dir=output_dir
                )

                best_model = {
                    "run_name": run_name,
                    "metrics": metrics,
                    "params": params,
                    "output_dir": output_dir
                }

                if os.path.exists(final_dir):
                    shutil.rmtree(final_dir)
                save_model(best_model, final_dir)
                print(f"✅ Modelo óptimo guardado en: {final_dir}")

            except Exception as e:
                print(f"❌ Error entrenando {model_name} en {dataset}: {e}")
                shutil.rmtree(output_dir, ignore_errors=True)

            shutil.rmtree(tmp_dir, ignore_errors=True)

def train_ml_missing():
    sys.path.append('../')
    from machine_learning.train_ml_missing import run_ml_training

    BASE_RESULTS_DIR = "../../optim_results/results_ml_missing"
    DATASETS = ["metilacion", "missing5", "missing10", "missing20", "missing30", "missing50"]

    best_params_ml_missing = {
        "metilacion": {
            "catboost": {
                "iterations": 50,
                "depth": 3,
                "learning_rate": 0.01
            },
            "xgboost": {
                "n_estimators": 100,
                "max_depth": 3,
                "learning_rate": 0.05
            },
            "histgb": {
                "max_iter": 50,
                "max_depth": 3,
                "min_samples_leaf": 5,
                "learning_rate": 0.1
            }
        },
        "missing5": {
            "catboost": {
                "iterations": 50,
                "depth": 5,
                "learning_rate": 0.05
            },
            "xgboost": {
                "n_estimators": 50,
                "max_depth": 3,
                "learning_rate": 0.05
            },
            "histgb": {
                "max_iter": 100,
                "max_depth": 3,
                "min_samples_leaf": 2,
                "learning_rate": 0.1
            }
        },
        "missing10": {
            "catboost": {
                "iterations": 50,
                "depth": 5,
                "learning_rate": 0.05
            },
            "xgboost": {
                "n_estimators": 50,
                "max_depth": 3,
                "learning_rate": 0.05
            },
            "histgb": {
                "max_iter": 50,
                "max_depth": 3,
                "min_samples_leaf": 2,
                "learning_rate": 0.01
            }
        },
        "missing20": {
            "catboost": {
                "iterations": 50,
                "depth": 5,
                "learning_rate": 0.01
            },
            "xgboost": {
                "n_estimators": 50,
                "max_depth": 3,
                "learning_rate": 0.05
            },
            "histgb": {
                "max_iter": 50,
                "max_depth": 3,
                "min_samples_leaf": 2,
                "learning_rate": 0.1
            }
        },
        "missing30": {
            "catboost": {
                "iterations": 50,
                "depth": 3,
                "learning_rate": 0.01
            },
            "xgboost": {
                "n_estimators": 50,
                "max_depth": 3,
                "learning_rate": 0.05
            },
            "histgb": {
                "max_iter": 100,
                "max_depth": 3,
                "min_samples_leaf": 5,
                "learning_rate": 0.1
            }
        },
        "missing50": {
            "catboost": {
                "iterations": 50,
                "depth": None,
                "learning_rate": 0.01
            },
            "xgboost": {
                "n_estimators": 50,
                "max_depth": 3,
                "learning_rate": 0.05
            },
            "histgb": {
                "max_iter": 100,
                "max_depth": 3,
                "min_samples_leaf": 5,
                "learning_rate": 0.1
            }
        }
    }

    FOLDS = 4
    SCALING_METHOD = "divided_by_100"

    for dataset in DATASETS:
        for model_name, params in best_params_ml_missing[dataset].items():
            print(f"\n🚀 Entrenando modelo óptimo: Dataset={dataset}, Modelo={model_name}")
            tmp_dir = os.path.join(BASE_RESULTS_DIR, dataset, model_name, "tmp")
            final_dir = os.path.join(BASE_RESULTS_DIR, dataset, model_name, "best_model")
            os.makedirs(tmp_dir, exist_ok=True)

            run_name = "best_run_manual"
            output_dir = os.path.join(tmp_dir, run_name)
            os.makedirs(output_dir, exist_ok=True)

            try:
                metrics = run_ml_training(
                    model_name=model_name,
                    model_params=params,
                    dataset=dataset,
                    n_folds=FOLDS,
                    scaling_method=SCALING_METHOD,
                    output_dir=output_dir
                )

                best_model = {
                    "run_name": run_name,
                    "metrics": metrics,
                    "params": params,
                    "output_dir": output_dir
                }

                if os.path.exists(final_dir):
                    shutil.rmtree(final_dir)
                save_model(best_model, final_dir)
                print(f"✅ Modelo óptimo guardado en: {final_dir}")

            except Exception as e:
                print(f"❌ Error entrenando {model_name} en {dataset}: {e}")
                shutil.rmtree(output_dir, ignore_errors=True)

            shutil.rmtree(tmp_dir, ignore_errors=True)

def main(mode=None):
    if mode is None:
        print("▶ Ejecutando todos los entrenamientos óptimos...")
        train_autoencoder_classifier()
        train_classifier()
        train_remasker_classifier()
        train_ml_models()
        train_ml_missing()
    elif mode == "autoencoder_classifier":
        train_autoencoder_classifier()
    elif mode == "classifier":
        train_classifier()
    elif mode == "remasker_classifier":
        train_remasker_classifier()
    elif mode == "ml_models":
        train_ml_models()
    elif mode == "ml_missing":
        train_ml_missing()
    else:
        raise ValueError(f"Modo desconocido: {mode}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ejecutar entrenamientos óptimos")
    parser.add_argument(
        "-m", "--mode",
        choices=[
            "autoencoder_classifier",
            "classifier",
            "remasker_classifier",
            "ml_models",
            "ml_missing"
        ],
        default=None,
        help="Modo de entrenamiento óptimo. Si no se especifica, ejecuta todos."
    )
    args = parser.parse_args()
    main(args.mode)

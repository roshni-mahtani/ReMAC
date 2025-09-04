import os
import pandas as pd

# Configuración base
BASE_DIR = "../../optim_results/results_remasker_classifier_best_mean_dec"
OUTPUT_DIR = "../../optim_results/tablas_remasker_best_mean_dec"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATASETS = ["metilacion", "missing5", "missing10", "missing20", "missing30", "missing50"]
EMBEDDING_SOURCES = ["encoder", "decoder"]
EMBEDDING_STRATEGIES = ["mean_wo_cls", "cls", "max_wo_cls"]
MASK_RATIOS = [10, 30, 50, 70]
METRICS = ["ACC", "SEN", "SPE", "PPV", "NPV", "F1", "AUC", "AUCPR"]

def extract_mean_std_row(csv_path):
    try:
        df = pd.read_csv(csv_path)
        # Buscar la fila que contiene "Mean" (puede variar según cómo exportaste los CSV)
        mean_row = df[df.iloc[:, 0].astype(str).str.contains("Mean", case=False, na=False)]
        if mean_row.empty:
            return ["NA"] * len(METRICS)
        return mean_row.iloc[0][1:1+len(METRICS)].tolist()
    except Exception as e:
        print(f"⚠️ Error leyendo {csv_path}: {e}")
        return ["NA"] * len(METRICS)

# Recorrer todas las combinaciones y generar 4 tablas
for source in EMBEDDING_SOURCES:
    for strategy in EMBEDDING_STRATEGIES:
        print(f"\n📊 Tabla para: {source.upper()} + {strategy.upper()}")

        # Estructura: dict[mask_ratio][dataset] = lista de métricas
        results_table = {mr: {} for mr in MASK_RATIOS}

        for mask in MASK_RATIOS:
            mask_dir = f"mask{mask}"
            for dataset in DATASETS:
                csv_path = os.path.join(
                    BASE_DIR,
                    dataset,
                    source,
                    strategy,
                    mask_dir,
                    "best_run",
                    "metrics",
                    "val_metrics_summary.csv"
                )
                row = extract_mean_std_row(csv_path)
                results_table[mask][dataset] = row

        # Crear DataFrame en bloques por mask_ratio
        full_table = pd.DataFrame()
        for mask in MASK_RATIOS:
            block_df = pd.DataFrame.from_dict(results_table[mask], orient="index", columns=METRICS)
            block_df.insert(0, "Dataset", block_df.index)
            block_df.insert(1, "MaskRatio", f"{mask}%")
            full_table = pd.concat([full_table, block_df], axis=0)

        # Guardar cada tabla en un CSV diferente
        output_name = f"summary_{source}_{strategy}.csv"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        full_table.to_csv(output_path, index=False)
        print(f"✅ Guardado: {output_name}")

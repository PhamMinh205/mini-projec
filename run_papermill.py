import papermill as pm
import os

NOTEBOOKS = [
    "preprocessing_and_eda.ipynb",
    "basket_preparation.ipynb",
    "apriori_modelling.ipynb",
    "fp_growth_modelling.ipynb",
    "compare_apriori_fpgrowth.ipynb",
    "clustering_from_rules.ipynb"
]

INPUT_DIR = "notebooks"
OUTPUT_DIR = "notebooks/runs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for nb in NOTEBOOKS:
    print(f"Running {nb} ...")
    pm.execute_notebook(
        os.path.join(INPUT_DIR, nb),
        os.path.join(OUTPUT_DIR, nb),
        kernel_name="python3"
    )

print("Pipeline executed successfully.")

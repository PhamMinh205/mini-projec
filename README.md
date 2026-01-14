# Mini Project – Customer Clustering from Association Rules

## Overview
This project performs customer segmentation based on association rules
mined from transaction data. Instead of traditional RFM-only clustering,
we transform strong association rules into behavioral features and apply
K-Means clustering.

## Pipeline
1. Data preprocessing & EDA
2. Basket preparation (boolean matrix)
3. Association rule mining (Apriori & FP-Growth)
4. Rule filtering by lift & confidence
5. Feature engineering from rules (+ optional RFM)
6. Customer clustering using K-Means
7. Visualization with PCA

## Project Structure

data/
├── raw/
├── processed/

src/
├── apriori_library.py
├── RuleBasedCustomerClusterer.py
├── cluster_library.py

notebooks/
├── preprocessing_and_eda.ipynb
├── basket_preparation.ipynb
├── apriori_modelling.ipynb
├── fp_growth_modelling.ipynb
├── compare_apriori_fpgrowth.ipynb
├── clustering_from_rules.ipynb


## How to Run
```bash
conda activate shopping_env
pip install -r requirements.txt
python run_papermill.py

Output

Association rules (CSV)

Customer cluster assignments

PCA visualizations

Course

Data Mining – FIT @ DNU
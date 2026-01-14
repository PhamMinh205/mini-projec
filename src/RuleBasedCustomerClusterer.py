import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


class RuleBasedCustomerClusterer:
    def __init__(self, top_k_rules=50, rule_weight="binary"):
        self.top_k_rules = top_k_rules
        self.rule_weight = rule_weight

    def load_rules(self, rules_df):
        rules_df = rules_df.sort_values("lift", ascending=False)
        self.rules = rules_df.head(self.top_k_rules).reset_index(drop=True)
        return self.rules

    def build_rule_features(self, transactions):
        customers = transactions["CustomerID"].unique()

        features = pd.DataFrame(
            0,
            index=customers,
            columns=[f"rule_{i}" for i in range(len(self.rules))]
        )

        basket = (
            transactions
            .groupby("CustomerID")["Description"]
            .apply(set)
        )

        for i, rule in self.rules.iterrows():
            antecedent = set(rule["antecedents"].split(","))

            for cid, items in basket.items():
                if antecedent.issubset(items):
                    if self.rule_weight == "binary":
                        features.loc[cid, f"rule_{i}"] = 1
                    elif self.rule_weight == "lift":
                        features.loc[cid, f"rule_{i}"] = rule["lift"]
                    elif self.rule_weight == "lift_conf":
                        features.loc[cid, f"rule_{i}"] = (
                            rule["lift"] * rule["confidence"]
                        )

        return features

    def build_rfm(self, df):
        snapshot = df["InvoiceDate"].max()

        rfm = (
            df.groupby("CustomerID")
            .agg({
                "InvoiceDate": lambda x: (snapshot - x.max()).days,
                "InvoiceNo": "count",
                "TotalPrice": "sum"
            })
            .rename(columns={
                "InvoiceDate": "Recency",
                "InvoiceNo": "Frequency",
                "TotalPrice": "Monetary"
            })
        )
        return rfm

    def build_final_features(self, rule_features, rfm=None):
        if rfm is not None:
            X = rule_features.join(rfm, how="left").fillna(0)
        else:
            X = rule_features.copy()

        scaler = StandardScaler()
        X[:] = scaler.fit_transform(X)
        return X

    def choose_k_by_silhouette(self, X, k_range=range(2, 11)):
        scores = {}
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42)
            labels = km.fit_predict(X)
            scores[k] = silhouette_score(X, labels)
        return scores

    def fit_kmeans(self, X, k):
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X)
        return labels

    def project_2d(self, X):
        pca = PCA(n_components=2, random_state=42)
        return pca.fit_transform(X)

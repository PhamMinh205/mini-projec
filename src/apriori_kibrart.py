import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules


class BasketPreparer:
    """
    Chuẩn bị dữ liệu basket (Customer × Item)
    """

    def __init__(self, min_item_freq=5):
        self.min_item_freq = min_item_freq

    def prepare_basket(self, df):
        """
        df cần có: InvoiceNo, CustomerID, Description
        """
        df = df.dropna(subset=["CustomerID", "Description"])

        # Lọc item ít xuất hiện
        item_freq = df["Description"].value_counts()
        valid_items = item_freq[item_freq >= self.min_item_freq].index
        df = df[df["Description"].isin(valid_items)]

        basket = (
            df.groupby(["InvoiceNo", "Description"])["Quantity"]
            .sum()
            .unstack()
            .fillna(0)
        )

        basket_bool = basket.applymap(lambda x: 1 if x > 0 else 0)
        return basket_bool


class RuleMiner:
    """
    Khai phá luật kết hợp bằng Apriori hoặc FP-Growth
    """

    def __init__(self, min_support=0.01, min_confidence=0.3, min_lift=1.0):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift

    def mine_apriori(self, basket_bool):
        freq = apriori(
            basket_bool,
            min_support=self.min_support,
            use_colnames=True
        )
        rules = association_rules(
            freq,
            metric="confidence",
            min_threshold=self.min_confidence
        )
        return self._filter_rules(rules)

    def mine_fpgrowth(self, basket_bool):
        freq = fpgrowth(
            basket_bool,
            min_support=self.min_support,
            use_colnames=True
        )
        rules = association_rules(
            freq,
            metric="confidence",
            min_threshold=self.min_confidence
        )
        return self._filter_rules(rules)

    def _filter_rules(self, rules):
        rules = rules[rules["lift"] >= self.min_lift].copy()
        rules["antecedents"] = rules["antecedents"].apply(
            lambda x: ",".join(sorted(list(x)))
        )
        rules["consequents"] = rules["consequents"].apply(
            lambda x: ",".join(sorted(list(x)))
        )
        return rules.sort_values("lift", ascending=False)

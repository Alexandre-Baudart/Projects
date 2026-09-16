from dataset import Dataset_
from analysis_utils import *

if __name__ == "__main__" :
    dataset = Dataset_(train_size=0.8)
    dataset.load_csv(data_path="data/heart_disease_uci.csv", dropped_cols=["id", "dataset"])
    dataset.binarize_target(old_target="num", new_target="target_binary", bin_threshold=0)

    df = dataset.get_dataframe()

    check_missing_values(df)
    check_count(df, target="target_binary")

    # heatmap_correlation(df, dropped_cols=["id", "num", "target_binary"] + [c for c in df.columns if "missing" in c])


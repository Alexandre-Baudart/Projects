from .dataset import Dataset_
from .analysis_utils import *

if __name__ == "__main__" :
    dataset = Dataset_(
        data_path="data/datasets/heart_disease_uci.csv",
        train_path="data/datasets/heart_disease_uci_train.csv",
        calib_path="data/datasets/heart_disease_uci_calib.csv",
        test_path="data/datasets/heart_disease_uci_test.csv",
        dropped_cols=["id", "dataset"],
        binarization_info={
            "old_target": "num",
            "new_target": "target_binary",
            "bin_threshold": 0
        },
        show_df_info=False
    )
    dataset.split_data_csv(train_size=0.8, calib_size=0.1, target="target_binary")

    df = dataset.get_original_df(binarize=True)

    # check_missing_values(df)
    # check_count(df, target="target_binary")

    # heatmap_correlation(df, dropped_cols=["id", "dataset"] + [c for c in df.columns if "missing" in c])
    # heatmap_correlation(df, dropped_cols=["id", "dataset", "target_binary"] + [c for c in df.columns if "missing" in c])


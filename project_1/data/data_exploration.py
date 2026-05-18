from project_utils import load_csv
from analysis_utils import check_missing_values, check_count, check_proportion, heatmap_correlation

if __name__ == "__main__" :
    df = load_csv("./data/creditcard.csv", show_info=False)

    # check_missing_values(df)
    # check_count(df, target="Class")
    # check_proportion(df, target="Class")

    # heatmap_correlation(df, dropped_cols=["Class"])


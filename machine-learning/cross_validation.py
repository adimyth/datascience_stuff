# taken from mlframework repo - https://github.com/abhishekkrthakur/mlframework
# single_col_regression stratified split - https://github.com/abhishekkrthakur/mlframework/pull/9/files
import pandas as pd
from sklearn import model_selection


class CrossValidation:
    def __init__(
        self,
        df,
        target_cols,
        shuffle,
        problem_type="binary_classification",
        multilabel_delimiter=",",
        num_folds=5,
        stratified_regression=False,
        stratified_regression_bins=20,
        random_state=42,
    ):
        self.dataframe = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.problem_type = problem_type
        self.num_folds = num_folds
        self.shuffle = shuffle
        self.stratified_regression = stratified_regression
        self.stratified_regression_bins = stratified_regression_bins
        self.random_state = random_state
        self.multilabel_delimiter = multilabel_delimiter

        if self.shuffle is True:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

        self.dataframe["kfold"] = -1

    def split(self):
        if self.problem_type in ("binary_classification", "multiclass_classification"):
            if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")
            target = self.target_cols[0]
            unique_values = self.dataframe[target].nunique()
            if unique_values == 1:
                raise Exception("Only one unique value found!")
            elif unique_values > 1:
                kf = model_selection.StratifiedKFold(n_splits=self.num_folds, shuffle=False)
                for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=self.dataframe[target].values)):
                    self.dataframe.loc[val_idx, "kfold"] = fold

        elif self.problem_type in ("single_col_regression", "multi_col_regression"):
            if self.num_targets != 1 and self.problem_type == "single_col_regression":
                raise Exception("Invalid number of targets for this problem type")
            if self.num_targets < 2 and self.problem_type == "multi_col_regression":
                raise Exception("Invalid number of targets for this problem type")
            kf = model_selection.KFold(n_splits=self.num_folds)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe)):
                self.dataframe.loc[val_idx, "kfold"] = fold

        # Similar - https://www.kaggle.com/youhanlee/stratified-sampling-for-regression-lb-1-4627#Target,-prediction-process
        elif self.problem_type=="single_col_regression" and self.stratified_regression:
            if self.num_targets != 1 and self.problem_type == "single_col_regression":
                raise Exception("Invalid number of targets for this problem type")
            target = self.target_cols[0]
            y = self.dataframe[target].values
            y_min = min(y)
            y_max = max(y)
            y_categorized = pd.cut(
                y,
                bins=range(y_min, y_max, 3),
                #bins=range(y_min, y_max, self.stratified_regression_bins),
                include_lowest=True,
                right=False,
                labels=range(y_min, y_max, self.stratified_regression_bins),
            )
            kf = model_selection.StratifiedKFold(n_splits=self.num_folds, shuffle=False)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=y_categorized)):
                self.dataframe.loc[val_idx, "kfold"] = fold

        elif self.problem_type.startswith("holdout_"):
            holdout_percentage = int(self.problem_type.split("_")[1])
            num_holdout_samples = int(len(self.dataframe) * holdout_percentage / 100)
            self.dataframe.loc[: len(self.dataframe) - num_holdout_samples, "kfold"] = 0
            self.dataframe.loc[len(self.dataframe) - num_holdout_samples :, "kfold"] = 1

        elif self.problem_type == "multilabel_classification":
            if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")
            targets = self.dataframe[self.target_cols[0]].apply(
                lambda x: len(str(x).split(self.multilabel_delimiter))
            )
            kf = model_selection.StratifiedKFold(n_splits=self.num_folds)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=targets)):
                self.dataframe.loc[val_idx, "kfold"] = fold
        else:
            raise Exception("Problem type not understood!")

        return self.dataframe


if __name__ == "__main__":
    import seaborn as sns
    import matplotlib.pyplot as plt
    df = pd.read_csv("../data/house_pricing/train.csv")
    problem_type = "single_col_regression"
    cv = CrossValidation(
        df,
        shuffle=True,
        target_cols=["SalePrice"],
        problem_type=problem_type,
        stratified_regression=True
    )
    df_split = cv.split()
    print("Split Data Sample")
    print(df_split.head())
    print(f"\nSize: {df_split.shape}")
    print("\nData Distribution by Fold")
    print(df_split.kfold.value_counts())
    print("\nTarget Distribution by Fold")
    if "_classification" in problem_type:
        print(df_split.groupby(by="kfold")["SalePrice"].value_counts())
    elif "_regression" in problem_type:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        sns.distplot(df_split[df_split["kfold"]==0]["SalePrice"], ax=ax1)
        ax1.set_title("Fold 0")
        sns.distplot(df_split[df_split["kfold"]==1]["SalePrice"], ax=ax2)
        ax2.set_title("Fold 1")
        sns.distplot(df_split[df_split["kfold"]==2]["SalePrice"], ax=ax3)
        ax3.set_title("Fold 2")
        sns.distplot(df_split[df_split["kfold"]==3]["SalePrice"], ax=ax4)
        ax4.set_title("Fold 3")
        plt.tight_layout()
        plt.show()

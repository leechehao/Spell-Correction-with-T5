"""
This module provides a script to split a large input csv file into train, validation,
and test datasets using stratified k-fold splitting.
"""

import argparse
import random

import pandas as pd
from sklearn.model_selection import StratifiedKFold


FOLD = "FOLD"
N_FOLDS = 10
N_TRAIN_FOLDS = 7
N_VAL_FOLDS = 1


def split_stratified_k_fold(
    df: pd.DataFrame,
    stratified_column: str,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Perform a stratified k-fold split on a pandas dataframe and add the fold number as a new column.

    Args:
        df (pd.DataFrame): The input pandas dataframe to be split.
        stratified_column (str): The column name based on which the stratified split should be performed.
        n_splits (int, optional): The number of folds to split the data into. Defaults to 5.

    Returns:
        pd.DataFrame: The input dataframe with the added `FOLD` column indicating the fold number of each data sample.
    """
    cv = StratifiedKFold(n_splits=n_splits)
    for n, (_, val_index) in enumerate(cv.split(df, df[stratified_column])):
        df.loc[val_index, FOLD] = int(n)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file to be split into train, validation, and test datasets.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the output file for the train dataset in LineSentence format. (one line = one sentence.)")
    parser.add_argument("--validation_file", type=str, required=True, help="Path to the output file for the validation dataset in LineSentence format. (one line = one sentence.)")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the output file for the test dataset in LineSentence format. (one line = one sentence.)")
    parser.add_argument("--input_column", type=str, required=True, help="Name of the column in the input CSV file that contains the data to be split into train, validation, and test datasets.")
    parser.add_argument("--stratified_column", type=str, required=True, help="Column used to perform stratified k-fold splitting of the data.")
    parser.add_argument("--seed", default=2330, type=int, help="Random seed used for data splitting.")
    args = parser.parse_args()

    # ===== Seed =====
    random.seed(args.seed)

    # ===== Load file =====
    df = pd.read_csv(args.input_file)

    # ===== Split data =====
    df = split_stratified_k_fold(
        df,
        stratified_column=args.stratified_column,
        n_splits=N_FOLDS,
    )
    folds = set(range(N_FOLDS))
    train_folds = set(random.sample(folds, N_TRAIN_FOLDS))
    val_folds = set(random.sample(folds - train_folds, N_VAL_FOLDS))
    test_folds = folds - train_folds - val_folds
    with open(args.train_file, "w") as train_file:
        train_file.write("\n".join(df[df[FOLD].isin(train_folds)][args.input_column].values))
    with open(args.validation_file, "w") as val_file:
        val_file.write("\n".join(df[df[FOLD].isin(val_folds)][args.input_column].values))
    with open(args.test_file, "w") as test_file:
        test_file.write("\n".join(df[df[FOLD].isin(test_folds)][args.input_column].values))


if __name__ == "__main__":
    main()

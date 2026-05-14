import numpy as np
import pandas as pd
import argparse
import csv


def print_cols():
    df = pd.read_csv("database/head_cancerreg.tsv", sep="\t", header=0)
    # df = pd.read_csv("database/head_nshds.tsv", sep="\t", header=0)

    l = list(df.columns)

    for i in l:
        print(f"'{i}': ")

    print("")

    for i in l:
        print(f"'{i}',")

    print(df)
    return


def get_diff_year(df, col1="birth_date", col2="death_date", col_out="life_span"):
    c1 = pd.to_datetime(df[col1], format="%Y-%m-%d")
    c2 = pd.to_datetime(df[col2], format="%Y-%m-%d")
    df[col_out] = c2.dt.year - c1.dt.year
    return df


def bin_year(df, col):
    c1 = pd.to_datetime(df[col], format="%Y-%m-%d")
    bins = list(range(0, 5000, 5))
    df[f"{col}_bin"] = pd.cut(c1.dt.year, bins)
    return df


def bin_span(df, col):
    bins = list(range(0, 200, 5))
    df[f"{col}_bin"] = pd.cut(df[col], bins)
    return df


def remove_unique_row_based_on_icd_code(df, col, id_col="mockid"):
    print(f"Before removing unique {col}: {len(df.index)}")

    m1 = df[id_col].isin(df.drop_duplicates(col, keep=False)[id_col])
    m2 = df.duplicated(id_col, keep=False)
    df = df[m1 ^ m2]

    print(f"After removing unique {col}: {len(df.index)}")

    return df


def remove_rows_small_occurrence(df, col, threshold=5):
    """
    Remove rows when the occurrence of a column value in
    the data frame is less than a certain number
    """
    print(f"Before removing unique {col}: {len(df.index)}")

    _df = df[col].value_counts(dropna=False)

    for val, count in _df.items():
        if count < threshold:
            df = df[df[col] != val]

    # df = df[df.groupby(col)[col].transform("count").ge(threshold)]

    print(f"After removing unique {col}: {len(df.index)}")

    return df


def remove_rows_small_occurrence_with_mockid(df, col, threshold=5):
    """
    Remove rows when the occurrence of a column value in
    the data frame is less than a certain number
    This function is different from remove_rows_small_occurrence:
        Ex:
            llkk_txt   mockid
            London     mock000005    1
                       mock000007    3
                       mock000008    2
            Method 1 => 6
            Method 2 => 3
    """
    print(f"Before removing unique {col}: {len(df.index)}")

    _df = df.groupby(col)["mockid"].nunique()

    for val, count in _df.items():
        if count < threshold:
            df = df[df[col] != val]

    # df = df[df.groupby(col)[col].transform("count").ge(threshold)]

    print(f"After removing unique {col}: {len(df.index)}")

    return df


def convert_categories_to_numbers(df, col):
    """
    pseudonymize data
    ["MO", "AB", NAN] -> [0,1, ""]
    """
    df[col] = pd.Categorical(df[col])
    df[col] = df[col].cat.codes
    df[col] = df[col].replace(-1, np.nan)
    return df


def main(args):
    workplace = (
        "database"
        if args.workplace == "local"
        else "/mnt/d/HOME/PREDICT/usrs/MV/00_example_data/data"
    )

    # read 3 files
    df_nshds = pd.read_csv(f"{workplace}/nshds.tsv", sep="\t", header=0)
    df_cancerreg = pd.read_csv(f"{workplace}/cancerreg.tsv", sep="\t", header=0)
    df_consent = pd.read_csv(f"{workplace}/predictconsent.tsv", sep="\t", header=0)

    # for consent file, we drop all columns except for ["mockid", "predict_cohort"]
    df_consent = df_consent[["mockid", "predict_cohort"]]

    # merge 3 files
    df = pd.merge(df_nshds, df_cancerreg, on="mockid", how="outer")
    df = pd.merge(df, df_consent, on="mockid", how="outer")

    print(df_nshds)
    print(df_cancerreg)
    print(df_consent)
    print(df)

    # keep only patients recruited in the PREDICT cohort
    df = df[df["predict_cohort"] == 1]

    # get diff in years (samp_span=age, qnr_span=age)
    df = get_diff_year(df, col1="birth_date", col2="samp_date", col_out="samp_span")
    df = get_diff_year(df, col1="birth_date", col2="qnr_date", col_out="qnr_span")
    df = get_diff_year(df, col1="birth_date", col2="death_date", col_out="life_span")
    df = get_diff_year(
        df, col1="birth_date", col2="vitalstatus_date", col_out="vital_span"
    )

    # compute age = either samp_span or qnr_span
    df["age"] = df["samp_span"]
    df.loc[df["age"].isnull(), "age"] = df["qnr_span"]

    # bin date in 5-year period
    for col in [
        "samp_date",
        "qnr_date",
        "birth_date",
        "dx_date",
        "vitalstatus_date",
        "death_date",
    ]:
        df = bin_year(df, col)

    # bin span in 5-year period
    for col in [
        "age",
        "samp_span",
        "qnr_span",
        "life_span",
        "vital_span",
    ]:
        df = bin_span(df, col)

    # print
    print(df["mockid"].value_counts(dropna=False))
    df1 = df["mockid"].value_counts(dropna=False)
    print(df1.value_counts())

    # remove patients that have very few occurrences (icd7_code, icd9_code, icdo2_code, icdo3_code)
    for col in ["llkk_txt", "icd7_code", "icd9_code", "icdo2_code", "icdo3_code"]:
        # df = remove_rows_small_occurrence(df, col, threshold=5)
        df = remove_rows_small_occurrence_with_mockid(df, col, threshold=5)

    # shuffle, reset and categoize in order of 123
    df = df.iloc[np.random.permutation(len(df))]
    df = df.reset_index(drop=True)
    df["mockid"] = pd.factorize(df["mockid"])[0] + 1

    # categorize sensitive columns
    for col in [
        "subproj",
        "vdc",
        "icd7_code",
        "icd9_code",
        "icdo2_code",
        "icdo3_code",
    ]:
        df = convert_categories_to_numbers(df, col)

    df["mockid"] = df["mockid"].apply(lambda x: f"id{x:06}")

    # round bmi
    df["bmi"] = df["bmi"].round()

    # drop sensitive or unimportant columns
    df = df.drop(
        columns=[
            # "mockid",
            "llkk",
            # "llkk_txt",
            "icd7_txt",
            "icd9_txt",
            "icdo2_txt",
            "icdo3_txt",
            "samp_date",
            "qnr_date",
            "birth_date",
            "dx_date",
            "vitalstatus_date",
            "death_date",
            "age",
            "samp_span",
            "qnr_span",
            "life_span",
            "vital_span",
        ]
    )

    # rename mockid
    df = df.rename(columns={"mockid": "id"})

    if args.workplace == "local":
        df.to_csv(
            "database/compiled.tsv", sep="\t", index=False, quoting=csv.QUOTE_NONE
        )
    else:
        df.to_csv("compiled.tsv", sep="\t", index=False, quoting=csv.QUOTE_NONE)

    print(df)


parser = argparse.ArgumentParser(add_help=False)

parser.add_argument(
    "-w", "--workplace", type=str, default="local", choices=["local", "bea"]
)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

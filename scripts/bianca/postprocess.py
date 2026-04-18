import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "3"  # 0 = all messages; 1 = filter out INFO; 2 = filter out INFO and WARNING; 3 = only errors
)

import glob
import argparse
import numpy as np
import pandas as pd
import shutil
from engine.datasets import get_dataset
import engine.utils.data_utils as data_utils

# Ignore all PerformanceWarnings from pandas
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def get_csv_files(directory):
    """
    Retrieves a list of all CSV files in a given directory.

    Args:
      directory: The path to the directory.

    Returns:
      A list of file paths to CSV files, or an empty list if none are found.
    """

    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    return csv_files


def get_subdirs_containing_string(directory, search_string):
    """
    Retrieves a list of subdirectories within a directory that contain a specified string.

    Args:
        directory: The path to the main directory.
        search_string: The string to search for in subdirectory names.

    Returns:
        A list of subdirectory paths that contain the search string, or an empty list.
    """
    matching_subdirs = []
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path) and search_string in subdir:
            matching_subdirs.append(subdir_path)
    return matching_subdirs


def convert_number_to_date(df):
    """
    Converts numerical values (rounded to integer) in DataFrame columns containing "date"
    back to datetime objects.

    Args:
        df (pd.DataFrame): The input DataFrame with numerical date representations.

    Returns:
        pd.DataFrame: The DataFrame with the "date" columns converted back to datetime.
    """
    reference_date = pd.Timestamp("1800-01-01")

    for column in df.columns:
        if "date" in column.lower():
            df[column] = df[column].apply(
                lambda days: (
                    pd.NaT
                    if pd.isna(days) or round(days) == -1
                    else reference_date + pd.Timedelta(days=round(days))
                )
            )
    return df


def create_inverse_filename(full_path):
    """
    Takes a full file path and returns a new path with "_inverse" inserted
    before the file extension.

    Args:
        full_path (str): The full path to the file (e.g., "/something/something/filename.csv").

    Returns:
        str: The new full path with "_inverse" (e.g., "/something/something/filename_inverse.csv").
    """
    directory, filename = os.path.split(full_path)
    name, extension = os.path.splitext(filename)
    new_filename = f"{name}_inverse{extension}"
    new_full_path = os.path.join(directory, new_filename)
    return new_full_path


def match_df_columns_and_types(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    exclude_list: list = [],
) -> pd.DataFrame:
    """
    Rearranges columns in df1 to match the order of df2 and sets the data type
    of each column in df1 to match the corresponding column in df2.
    Adds missing columns from df2 to df1 (filled with NaN) and drops columns
    from df1 not present in df2.

    Args:
        df1 (pd.DataFrame): The DataFrame to modify.
        df2 (pd.DataFrame): The reference DataFrame for column order and types.

    Returns:
        pd.DataFrame: A new DataFrame based on df1 with columns and types
                      matching df2.
    """
    # Get the desired column order from df2
    desired_column_order = df2.columns.tolist()

    # Identify columns in df1 that are not in df2
    cols_to_drop_from_df1 = [
        col for col in df1.columns if col not in desired_column_order
    ]

    # Identify columns in df2 that are not in df1
    cols_to_add_to_df1 = [col for col in desired_column_order if col not in df1.columns]

    # Create a copy of df1 to avoid modifying the original DataFrame in place
    df1_modified = df1.copy()

    # Drop columns from df1 that are not in df2
    if cols_to_drop_from_df1:
        print(f"Dropping columns from df1 not in df2: {cols_to_drop_from_df1}")
        df1_modified = df1_modified.drop(columns=cols_to_drop_from_df1)

    # Add columns to df1 that are in df2 but not in df1
    if cols_to_add_to_df1:
        print(f"Adding columns to df1 from df2: {cols_to_add_to_df1}")
        for col in cols_to_add_to_df1:
            # Add the column, initially filled with NaN
            df1_modified[col] = np.nan
            # Set the dtype of the newly added column to match df2 immediately
            # This is important if the target dtype is non-numeric (like object or category)
            if (
                col in df2.columns
            ):  # Should always be true based on cols_to_add_to_df1 logic
                target_dtype = df2[col].dtype
                try:
                    df1_modified[col] = df1_modified[col].astype(target_dtype)
                except TypeError as e:
                    print(
                        f"Warning: Could not set initial dtype for new column '{col}' to {target_dtype}. Reason: {e}"
                    )
                    print("Attempting type conversion after reindexing.")

    # Reindex df1_modified to match the column order of df2
    # This will also handle adding columns if they weren't added explicitly above
    # (though explicit adding helps set initial dtypes for non-numeric)
    df1_modified = df1_modified.reindex(columns=desired_column_order)

    # Set the data type of each column in df1_modified to match df2
    print("Matching data types of columns...")
    for col in desired_column_order:
        if col in exclude_list:
            continue

        if col in df1_modified.columns and col in df2.columns:
            target_dtype = df2[col].dtype
            current_dtype = df1_modified[col].dtype

            if current_dtype != target_dtype:
                # print(
                #     f"  Converting column '{col}' from {current_dtype} to {target_dtype}"
                # )
                try:
                    # Use errors='coerce' to turn values that cannot be converted into NaN
                    # This is important for robustness, e.g., converting strings to numbers
                    df1_modified[col] = df1_modified[col].astype(
                        target_dtype, errors="ignore"
                    )
                except Exception as e:
                    # Catching a general exception here as astype can raise various errors
                    print(
                        f"  Warning: Could not convert column '{col}' to {target_dtype}. Reason: {e}"
                    )
                    print("  Values that failed conversion might be replaced by NaN.")
        elif col not in df1_modified.columns:
            pass
            # print(
            #     f"  Column '{col}' from df2 was added but not found in df1_modified after reindexing. This is unexpected."
            # )

    return df1_modified


def copy_preprocessed(dir):
    is_copy = True
    if "tabsyn" in dir:
        csv_files = get_csv_files(dir)
        for csv_file in csv_files:
            if "preprocessed.csv" in csv_file:
                is_copy = False

    if is_copy:
        src = f"database/dataset/{args.dataset}/temp/preprocessed.csv"
        dst = f"{dir}/preprocessed.csv"
        print(f">> copy {src} to {dst}")
        shutil.copyfile(src, dst)


def fix_pandas_writing(df, csv_file):
    real_cols = list(pd.read_csv(csv_file, sep="\t", nrows=0).columns)
    df = df.loc[:, df.columns.isin(real_cols)]

    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df.dropna(how="any")
    df = df.reset_index(drop=True)
    return df


def _legacy_do_postprocess(args):
    dirs = get_subdirs_containing_string(args.folder, f"{args.dataset}-")
    if args.arch is not None:
        dirs = [dir for dir in dirs if args.arch in dir]
    dirs.sort()

    D = get_dataset(args.dataset)
    label_encoder_path = D.label_encoder_path

    for dir in dirs:
        print()
        print()
        print()
        print("-" * 100)
        print(f">> processing {dir}")
        print("-" * 100)
        print()
        print()
        print()

        copy_preprocessed(dir)

        csv_files = get_csv_files(dir)
        print(get_csv_files)

        """
        flow
            forward
                convert_date_to_number -> preprocess_missing (cont.) -> encode binary (cat.) -> pca -> encode (classification)
            backward
                decode (classification) -> pca > decode binary (cat.) -> preprocess_missing (cont.) -> convert_number_to_date
        """
        for csv_file in csv_files:
            if "_inverse.csv" in csv_file:
                continue

            print()
            print()
            print()
            print("-" * 100)
            print(f"    >> processing {csv_file}")
            print("-" * 100)
            print()
            print()
            print()

            # 1) Read with Python engine and skip bad lines --  classic “ragged row” error
            df_fake = pd.read_csv(
                csv_file,
                sep="\t",
                header=0,
                index_col=0,
                engine="python",  # slower but more flexible
                on_bad_lines="skip",  # drop any line that doesn't match the header
                quoting=3,  # csv.QUOTE_NONE to avoid stray-quote issues
            )

            print()
            print(">> fixing pandas writing")
            print()
            df_fake = fix_pandas_writing(df_fake, csv_file)
            df_fake_inverse = data_utils.inverse_transform(
                df_fake, label_encoder_path=label_encoder_path
            )
            inverse_target_column = df_fake_inverse[D.target].copy()

            print("df_inverse label_encoder_path")
            print(df_fake_inverse)

            df_data_pca = D.data_pca
            df_data_original = D.data_original

            print()
            print(">> inversing pca")
            print()
            # inverve pca
            (columns_to_pca, cont_cols, discrete_cols, [], binary_cols) = (
                D.data_original_columns
            )
            columns_to_keep_pca = [
                col for col in df_data_pca.columns if col != D.target
            ]
            X_pca = df_fake_inverse.drop(columns=[D.target]).copy().to_numpy()
            df_fake_inverse = D.pca.inverse_transform(
                X_pca,
                df_data_pca[columns_to_keep_pca].copy(),
                cont_cols,
                discrete_cols,
                [],
                binary_cols,
            )

            print()
            print(">> inversing binary")
            print()

            # inverse binary
            df_fake_inverse = D.binary_encoder.inverse_transform(df_fake_inverse)

            print()
            print(">> inversing continuous")
            print()

            # inverse preprocess_missing continuous
            list_column_missing = []
            for col in df_fake_inverse.columns:
                flag = f"{col}_missing"
                if flag not in df_fake_inverse.columns:
                    # no flag—nothing to undo
                    continue

                # Where the flag is 1, restore NaN:
                list_column_missing.append(flag)
                df_fake_inverse.loc[df_fake_inverse[flag].astype(str) == "1", col] = (
                    np.nan
                )

            df_fake_inverse = df_fake_inverse.drop(columns=list_column_missing)

            print()
            print(">> inversing date")
            print()
            df_fake_inverse = convert_number_to_date(df_fake_inverse)

            # inverse preprocess_missing discrete
            df_fake_inverse = df_fake_inverse.applymap(
                lambda x: np.nan if str(x) == "-1" else x
            )

            df_fake_inverse = pd.concat(
                [df_fake_inverse, inverse_target_column], axis=1
            )

            df_fake_inverse = match_df_columns_and_types(
                df_fake_inverse,
                df_data_original,
                exclude_list=[D.target],
            )

            print("df_fake_inverse final")
            print(df_fake_inverse)

            path = create_inverse_filename(csv_file)
            print(csv_file)
            print(path)
            df_fake_inverse.to_csv(
                path,
                sep="\t",
                encoding="utf-8",
            )

            print(df_data_pca)
            print(df_fake_inverse)


def do_postprocess(args):
    dirs = get_subdirs_containing_string(args.folder, f"{args.dataset}-")
    if args.arch is not None:
        dirs = [dir for dir in dirs if args.arch in dir]
    dirs.sort()

    D = get_dataset(args.dataset)
    label_encoder_path = D.label_encoder_path
    pipeline = D.pipeline

    for dir in dirs:
        print()
        print()
        print()
        print("-" * 100)
        print(f">> processing {dir}")
        print("-" * 100)
        print()
        print()
        print()

        copy_preprocessed(dir)

        csv_files = get_csv_files(dir)
        print(get_csv_files)

        """
        flow
            forward
                convert_date_to_number -> preprocess_missing (cont.) -> encode binary (cat.) -> pca -> encode (classification)
            backward
                decode (classification) -> pca > decode binary (cat.) -> preprocess_missing (cont.) -> convert_number_to_date
        """
        for csv_file in csv_files:
            if "_inverse.csv" in csv_file:
                continue

            print()
            print()
            print()
            print("-" * 100)
            print(f"    >> processing {csv_file}")
            print("-" * 100)
            print()
            print()
            print()

            # 1) Read with Python engine and skip bad lines --  classic “ragged row” error
            df_fake = pd.read_csv(
                csv_file,
                sep="\t",
                header=0,
                index_col=0,
                engine="python",  # slower but more flexible
                on_bad_lines="skip",  # drop any line that doesn't match the header
                quoting=3,  # csv.QUOTE_NONE to avoid stray-quote issues
            )

            print()
            print(">> fixing pandas writing")
            print()
            df_fake = fix_pandas_writing(df_fake, csv_file)

            print()
            print(">> data_utils.inverse_transform")
            print(f"     load {label_encoder_path}")
            print()
            df_fake_inverse = data_utils.inverse_transform(
                df_fake, label_encoder_path=label_encoder_path
            )

            print("df_inverse label_encoder_path")
            print(df_fake_inverse)

            print()
            print(">> pipeline.inverse_transform")
            print()
            df_fake_inverse = pipeline.inverse_transform(df_fake_inverse)

            print("df_inverse pipeline.inverse_transform")
            print(df_fake_inverse)

            df_data_original = D.data_original
            df_fake_inverse = match_df_columns_and_types(
                df_fake_inverse,
                df_data_original,
                exclude_list=[D.target],
            )

            print("df_fake_inverse final")
            print(df_fake_inverse)

            path = create_inverse_filename(csv_file)
            print(csv_file)
            print(path)
            df_fake_inverse.to_csv(
                path,
                sep="\t",
                encoding="utf-8",
            )


parser = argparse.ArgumentParser(add_help=False)
# generate subsets of a pandas DataFrame by subsampling rows and shuffling columns before sampling

parser.add_argument(
    "--folder",
    # default="_database/gan_optimize",
    default="database/gan",
    type=str,
)
parser.add_argument(
    "--dataset",
    # default="biobank_phase3_cancer_bio_all",
    default="biobank_phase3_dummy",
    type=str,
)
parser.add_argument(
    "--arch",
    default=None,
    type=str,
)


args = parser.parse_args()


# Example Usage:
if __name__ == "__main__":
    do_postprocess(args)

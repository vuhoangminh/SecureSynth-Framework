import os
import glob
import argparse
import time
import pandas as pd
import numpy as np

from engine.datasets import get_dataset
import engine.utils.data_utils as data_utils

from termcolor import colored


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def plot_column_distributions(df, output_pdf="column_distributions.pdf"):
    """
    Iterates through each column of a pandas DataFrame, computes the percentage of NaNs,
    and plots the distribution of non-NaN values. The plot title shows the percentage
    of NaNs (formatted to four decimal places). The plot is saved to a single PDF.

    Args:
        df (pd.DataFrame): The input pandas DataFrame.
        output_pdf (str, optional): The name of the output PDF file.
            Defaults to "column_distributions.pdf".
    """
    pdf = PdfPages(output_pdf)
    num_rows = len(df)

    for col in df.columns:
        series = df[col]
        percent_nan = series.isna().mean() * 100
        percent_nan_str = f"{percent_nan:.4f}%"

        # Only plot non-NaN values.
        non_nan_series = series.dropna()

        # Identify column type.
        is_numeric = False
        is_text = False
        is_date = pd.api.types.is_datetime64_any_dtype(series)

        # Explicitly check if series contains booleans, even if dtype is object.
        if pd.api.types.is_bool_dtype(series) or all(
            isinstance(x, bool) for x in non_nan_series
        ):
            is_numeric = True
            # Convert True -> 1, False -> 0, then to float32
            non_nan_series = non_nan_series.astype(int).astype("float32")
        elif not is_date:
            try:
                # Try converting to numeric.
                pd.to_numeric(non_nan_series)
                is_numeric = True
            except Exception:
                # If conversion fails, check if all non-NaN items are strings.
                if non_nan_series.apply(lambda x: isinstance(x, str)).all():
                    try:
                        pd.to_datetime(non_nan_series)
                        is_date = True
                    except Exception:
                        is_text = True

        plt.figure(figsize=(10, 6))
        plt.title(f"Distribution of Column: {col} (NaN: {percent_nan_str})")
        plt.xlabel(col)
        plt.ylabel("Frequency")

        if is_numeric:
            # Convert to numeric explicitly, then drop any remaining NaNs.
            numeric_series = pd.to_numeric(non_nan_series, errors="coerce").dropna()
            plt.hist(numeric_series, bins=20, edgecolor="black")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
        elif is_text:
            if not non_nan_series.empty:
                value_counts = non_nan_series.value_counts()
                if len(value_counts) > 0:
                    value_counts.plot(kind="bar")
                    plt.ylabel("Count")
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                else:
                    plt.text(
                        0.5, 0.5, "No non-NaN values to plot.", ha="center", va="center"
                    )
            else:
                plt.text(
                    0.5, 0.5, "No non-NaN values to plot.", ha="center", va="center"
                )
        elif is_date:
            try:
                date_series = pd.to_datetime(non_nan_series, errors="raise")
                if not date_series.empty:
                    date_series.hist()
                    plt.ylabel("Frequency")
                else:
                    plt.text(
                        0.5,
                        0.5,
                        "No non-NaN date values to plot.",
                        ha="center",
                        va="center",
                    )
            except Exception:
                value_counts = non_nan_series.value_counts()
                if len(value_counts) > 0:
                    value_counts.plot(kind="bar")
                    plt.ylabel("Count")
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                else:
                    plt.text(
                        0.5, 0.5, "No non-NaN values to plot.", ha="center", va="center"
                    )
        else:
            plt.text(
                0.5,
                0.5,
                f"Column '{col}' could not be identified for plotting.",
                ha="center",
                va="center",
            )

        pdf.savefig()
        plt.close()

    pdf.close()
    print(f"All column distributions have been saved to '{output_pdf}'")


def is_valid_number(value):
    """Checks if a value is a valid number, including scientific notation."""
    if pd.api.types.is_numeric_dtype(type(value)):
        return True
    if isinstance(value, str):
        try:
            float(value)
            return True
        except ValueError:
            return False
    return False


def analyze_dataframe_columns(df):
    """
    Analyzes each column of a pandas DataFrame for format conflicts (numbers and text)
    and prints the describe() output with sample conflicts.

    Args:
        df (pd.DataFrame): The input pandas DataFrame.
    """
    for col in df.columns:
        series = df[col]
        non_nan_series = series.dropna()

        time.sleep(0.1)
        print()
        print()
        print()
        print()
        print()
        print("-" * 100)
        print(f"Analyzing column: {col}")
        print("-" * 100)

        # Check for format conflicts (numbers and text)
        all_are_numbers = all(is_valid_number(value) for value in non_nan_series)
        has_text = any(
            isinstance(value, str) and not is_valid_number(value)
            for value in non_nan_series
        )
        has_other = any(
            not is_valid_number(value) and not isinstance(value, str)
            for value in non_nan_series
        )

        conflict_message = None
        if all_are_numbers:
            pass  # No conflict if all non-NaN values are numbers
        elif has_text and all_are_numbers:
            # This case shouldn't happen with the current logic, but for robustness
            pass
        elif has_text and any(is_valid_number(value) for value in non_nan_series):
            conflict_message = "Potential format conflict: Column contains both numbers (including scientific) and non-numeric text."
            numeric_samples = [
                value for value in non_nan_series if is_valid_number(value)
            ][:3]
            text_samples = [
                value
                for value in non_nan_series
                if isinstance(value, str) and not is_valid_number(value)
            ][:3]
            if conflict_message:
                print(colored(conflict_message, "red"))
                if numeric_samples:
                    print(f"  Numeric Samples: {numeric_samples}")
                if text_samples:
                    print(f"  Text Samples: {text_samples}")
        elif has_other and any(is_valid_number(value) for value in non_nan_series):
            conflict_message = "Potential format conflict: Column contains numbers (including scientific) and other non-string types."
            numeric_samples = [
                value for value in non_nan_series if is_valid_number(value)
            ][:3]
            other_samples = {}
            for value in non_nan_series:
                if not is_valid_number(value) and not isinstance(value, str):
                    val_type = type(value)
                    if val_type not in other_samples:
                        other_samples[val_type] = []
                    if len(other_samples[val_type]) < 3:
                        other_samples[val_type].append(value)
            if conflict_message:
                print(colored(conflict_message, "red"))
                if numeric_samples:
                    print(f"  Numeric Samples: {numeric_samples}")
                for type_name, samples in other_samples.items():
                    print(f"  {type_name.__name__} Samples: {samples}")
        elif has_other and has_text:
            conflict_message = "Potential format conflict: Column contains non-numeric text and other non-string types."
            text_samples = [
                value
                for value in non_nan_series
                if isinstance(value, str) and not is_valid_number(value)
            ][:3]
            other_samples = {}
            for value in non_nan_series:
                if not is_valid_number(value) and not isinstance(value, str):
                    val_type = type(value)
                    if val_type not in other_samples:
                        other_samples[val_type] = []
                    if len(other_samples[val_type]) < 3:
                        other_samples[val_type].append(value)
            if conflict_message:
                print(colored(conflict_message, "red"))
                if text_samples:
                    print(f"  Text Samples: {text_samples}")
                for type_name, samples in other_samples.items():
                    print(f"  {type_name.__name__} Samples: {samples}")
        elif non_nan_series.dtype == "object":
            # Further check for mixed non-string types if not all are numbers
            if not all_are_numbers:
                types = set(type(item) for item in non_nan_series)
                if len(types) > 1:
                    conflict_message = "Potential format conflict: Column with 'object' dtype contains a mix of types (where not all are valid numbers)."
                    print(colored(conflict_message, "red"))
                    type_samples = {}
                    for value in non_nan_series:
                        val_type = type(value)
                        if val_type not in type_samples:
                            type_samples[val_type] = []
                        if len(type_samples[val_type]) < 3:
                            type_samples[val_type].append(value)
                    for type_name, samples in type_samples.items():
                        print(f"  {type_name.__name__} Samples: {samples}")

        # Print describe()
        print("\nColumn Description:")
        print(series.describe())


parser = argparse.ArgumentParser(add_help=False)
# generate subsets of a pandas DataFrame by subsampling rows and shuffling columns before sampling
parser.add_argument(
    "--dataset",
    default="nshds_cancer_bio_all.tsv",
    type=str,
)
parser.add_argument(
    "--folder",
    # default="_database/gan_optimize",
    default="database/dataset/daniel",
    type=str,
)


args = parser.parse_args()

# dirs = get_subdirs_containing_string(args.folder, args.dataset)

csv_file = os.path.join(args.folder, args.dataset)

# df = pd.read_csv(csv_file, sep="\t", header=0, index_col=0)
df = pd.read_csv(csv_file, sep="\t", header=0)
print(df)

print()
print()
print()
print()
print()
analyze_dataframe_columns(df)


print()
print()
print()
print()
print()
print()
print()
print()
print()
print()
filename = args.dataset.replace(".tsv", ".pdf")
plot_column_distributions(df, output_pdf=filename)

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "3"  # 0 = all messages; 1 = filter out INFO; 2 = filter out INFO and WARNING; 3 = only errors
)

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import argparse
from engine.datasets import get_dataset
import engine.utils.data_utils as data_utils
from engine.analysis import analyze_fake_vs_real_single_run_pdf
from engine.dataset_helper.preprocessing import MissingValueEncoder

# Set the color scheme
colors = {"Original": "#BC3C29", "Synthetic": "#0072B5"}


def compare_distributions_to_pdf(
    df, df_fake, output_pdf="comparison.pdf", unique_threshold=15
):
    """
    Compares distributions for each column from two DataFrames and saves the plots to a PDF.
    For continuous variables, overlays KDE plots.
    For discrete/categorical variables, shows a grouped bar plot of value counts.
    """
    pdf = PdfPages(output_pdf)
    num_cols = len(df.columns)

    for i, col in enumerate(df.columns):
        print(f">> processing {i+1}/{num_cols}: {col}")
        # if (
        #     "date" in col
        #     or "rand36" in col
        #     or "qol" in col
        #     or "issi" in col
        #     or "cage" in col
        # ):
        #     continue

        percent_nan = df[col].isna().mean() * 100
        percent_nan_str = f"{percent_nan:.4f}%"

        series_orig = df[col].dropna()
        series_fake = df_fake[col].dropna()

        is_date = pd.api.types.is_datetime64_any_dtype(df[col])
        is_numeric = False
        is_text = False

        if not is_date:
            if pd.api.types.is_bool_dtype(df[col]):
                is_numeric = True
                series_orig = series_orig.astype("int").astype("float")
                series_fake = series_fake.astype("int").astype("float")
            else:
                try:
                    pd.to_numeric(series_orig)
                    is_numeric = True
                except Exception:
                    if series_orig.apply(lambda x: isinstance(x, str)).all():
                        try:
                            pd.to_datetime(series_orig)
                            is_date = True
                        except Exception:
                            is_text = True
                    else:
                        is_text = True

        plt.figure(figsize=(12, 6))
        plt.title(f"Distribution: {col} (NaN: {percent_nan_str})")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        ax = plt.gca()

        if is_numeric and not is_date:
            if series_orig.nunique() < unique_threshold:
                is_numeric = False
                is_text = True
            else:
                sns.kdeplot(
                    series_orig, label="Original", color=colors["Original"], ax=ax
                )
                sns.kdeplot(
                    series_fake, label="Synthetic", color=colors["Synthetic"], ax=ax
                )
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        if is_text or is_date:
            vc_orig = series_orig.value_counts(sort=False)
            vc_fake = series_fake.value_counts(sort=False)

            all_cats = sorted(set(vc_orig.index).union(vc_fake.index))
            orig_vals = [vc_orig.get(cat, 0) for cat in all_cats]
            fake_vals = [vc_fake.get(cat, 0) for cat in all_cats]
            x = np.arange(len(all_cats))

            width = 0.4
            ax.bar(
                x - width / 2,
                orig_vals,
                width,
                label="Original",
                color=colors["Original"],
            )
            ax.bar(
                x + width / 2,
                fake_vals,
                width,
                label="Synthetic",
                color=colors["Synthetic"],
            )

            ax.set_xticks(x)
            ax.set_xticklabels(all_cats, rotation=45, ha="right")
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            plt.tight_layout()

        pdf.savefig()
        plt.close()

    pdf.close()
    print(f"All column distributions have been saved to '{output_pdf}'")


def process_folder(folder_path):
    """
    In a given folder, check for 'preprocessed_inverse.csv' and any file matching 'fake_*_inverse.csv'.
    If both exist, read them as DataFrames with tab-separated values,
    then generate a PDF using compare_distributions_to_pdf with the folder name.
    """
    print(f"    >> processing {folder_path}...")
    preprocessed_file = os.path.join(folder_path, "preprocessed_inverse.csv")
    fake_pattern = os.path.join(folder_path, "fake_*_inverse.csv")
    fake_files = glob.glob(fake_pattern)

    if not os.path.exists(preprocessed_file) or not fake_files:
        print(f"Skipping folder '{folder_path}': Missing required CSV files.")
        return

    # For this example, choose the first matching fake file.
    fake_file = fake_files[0]

    try:
        df_real = pd.read_csv(preprocessed_file, sep="\t", header=0, index_col=0)
        df_fake = pd.read_csv(fake_file, sep="\t", header=0, index_col=0)

        # # for old and fail phase 3
        # df = MissingValueEncoder().set_all_neg1_to_nan(df)
        # df_fake = MissingValueEncoder().set_all_neg1_to_nan(df_fake)
    except Exception as e:
        print(f"Error reading CSV files in folder '{folder_path}': {e}")
        return

    # Use folder name as the PDF filename:
    folder_name = os.path.basename(os.path.normpath(folder_path))
    output_pdf = os.path.join(folder_path, f"{folder_name}.pdf")

    print()
    print()
    print()
    print("-" * 100)
    print(f">> processing {folder_path}")
    print(f"    >> saving plots to {output_pdf}")
    print("-" * 100)
    print()
    print()
    print()

    # compare_distributions_to_pdf(df, df_fake, output_pdf=output_pdf)
    excluded_keywords = ["date", "rand36", "qol", "issi", "cage"]
    list_columns = [
        col
        for col in df_real.columns
        if not any(key in col.lower() for key in excluded_keywords)
    ]

    if "cancer_bio_all" not in folder_path:
        included_keywords = [
            "age",
            "height",
            "weight",
            "bmi",
            "waist",
            "cholesterol",
            "gluc",
            "smoke",
            "alco",
            "active",
            "accessdate",
            "sm_duration",
            "sn_duration",
            "triglycerides",
            "education",
            "health_status_year",
            "cardio",
            "cancer",
            "icd9_code",
        ]
    else:
        included_keywords = [
            "age",
            "height",
            "weight",
            "bmi",
            "waist",
            "cholesterol",
            # "gluc",
            "smoke",
            # "alco",
            "active",
            "accessdate",
            "sm_duration",
            "sn_duration",
            "triglycerides",
            "education",
            "health_status_year",
            "cardio",
            "cancer",
            "icd9_code",
            "total_c",
            "dha",
            "albumin",
            "creatinine",
        ]

    list_columns = [
        col
        for col in df_real.columns
        if any(key in col.lower() for key in included_keywords)
    ]

    list_discrete_columns = [
        "cardio",
        "cancer",
        "icd9_code",
        "sex",
        "gender",
    ]

    if "cancer" in df_fake.columns:
        violin_specs = [
            (
                "cancer",
                "sm_duration",
                "Cancer",
                "Smoking Duration",
            ),
            (
                "cancer",
                "age",
                "Cancer",
                "Age",
            ),
            (
                "cancer",
                "bmi",
                "Cancer",
                "BMI",
            ),
        ]
    else:
        violin_specs = []

    analyze_fake_vs_real_single_run_pdf(
        df_real,
        df_fake,
        list_columns=list_columns,
        list_discrete_columns=list_discrete_columns,
        violin_specs=violin_specs,
        output_pdf=output_pdf,
    )


def process_root(args):
    """
    Recursively process each folder in root_dir.
    """
    dirs = os.listdir(args.folder)
    dirs.sort()

    print(dirs)

    # for dir in dirs:
    #     if args.dataset in dir:
    #         process_folder(dir)

    for dirpath, dirnames, filenames in os.walk(args.folder):
        if args.dataset in dirpath:
            process_folder(dirpath)


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
    default="biobank_phase3_dummy_pca",
    type=str,
)
args = parser.parse_args()


# Example Usage:
if __name__ == "__main__":
    process_root(args)

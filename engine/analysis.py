import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mcolors
from engine.config import config


TITLE_FONTSIZE = 30
# XYTICKS_LABELS_FONTSIZE = 25
XYTICKS_PARAMS_FONTSIZE = 20
LEGEND_FONTSIZE = 20


def adjust_color(color_name, percentage):
    """Adjusts the color intensity based on the given percentage."""
    if color_name not in config.COLORS:
        raise ValueError(f"Color '{color_name}' not defined.")

    # Get the original color (as RGB values)
    original_color = config.COLORS[color_name]

    # Calculate the mixed color
    mix_percentage = percentage / 100
    mixed_color = tuple(
        (mix_percentage * c + (1 - mix_percentage)) for c in original_color
    )

    return mixed_color


def compare_statistics_old(d):
    txt = "{:<30s}".format("")
    for key, df in d.items():
        txt = txt + "{:>25s}".format(key)

    d_describe = {}
    for key, df in d.items():
        d_describe[key] = df.describe()

    print(txt)
    print("-" * 120)
    for col in df:
        col_txt = "{:<30s}".format(col)
        for key, df_describe in d_describe.items():
            mean = df_describe[col]["mean"]
            std = df_describe[col]["std"]
            df_txt = "{:.4f} ({:.4f})".format(mean, std)
            col_txt = col_txt + "{:>25s}".format(df_txt)
        print(col_txt)


def compare_statistics(d):
    txt = "{:<30s}".format("")
    for key, df in d.items():
        txt = txt + "{:>25s}".format(key)

    d_describe = {}
    for key, df in d.items():
        d_describe[key] = df.describe()

    # print(txt)
    # print("-" * 90)

    p = {}
    p["Column"] = []
    for key, _ in d.items():
        p[key] = []

    for col in df:
        p["Column"].append(col)
        for key, df_describe in d_describe.items():
            mean = df_describe[col]["mean"]
            std = df_describe[col]["std"]
            p[key].append("{:.4f} ({:.4f})".format(mean, std))

    df = pd.DataFrame(p)

    return df


def compare_dataframe_distributions_sequential(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df1_name: str = "DataFrame 1",
    df2_name: str = "DataFrame 2",
):
    """
    Plots distributions of common numerical columns for two DataFrames side-by-side,
    showing each column's comparison plot immediately after processing.

    Args:
        df1 (pd.DataFrame): The first pandas DataFrame.
        df2 (pd.DataFrame): The second pandas DataFrame.
        df1_name (str): A name for the first DataFrame to use in titles. Defaults to 'DataFrame 1'.
        df2_name (str): A name for the second DataFrame to use in titles. Defaults to 'DataFrame 2'.
    """
    """
    Plots correlation matrices and distributions of numerical columns for two DataFrames side-by-side.

    Args:
        df1 (pd.DataFrame): The first pandas DataFrame.
        df2 (pd.DataFrame): The second pandas DataFrame.
        df1_name (str): A name for the first DataFrame to use in titles. Defaults to 'DataFrame 1'.
        df2_name (str): A name for the second DataFrame to use in titles. Defaults to 'DataFrame 2'.
    """

    # --- 1. Plot Correlation Matrices Side-by-Side ---

    # Select only numerical columns
    df1_numerical = df1.select_dtypes(include=np.number)
    df2_numerical = df2.select_dtypes(include=np.number)

    # Calculate correlation matrices
    correlation_matrix1 = df1_numerical.corr()
    correlation_matrix2 = df2_numerical.corr()

    # Create a figure with two subplots side-by-side
    fig1, axes1 = plt.subplots(1, 2, figsize=(18, 8))  # 1 row, 2 columns

    # Plot heatmap for df1
    sns.heatmap(
        correlation_matrix1,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        ax=axes1[0],
    )
    axes1[0].set_title(f"Correlation Matrix ({df1_name})")

    # Plot heatmap for df2
    sns.heatmap(
        correlation_matrix2,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        ax=axes1[1],
    )
    axes1[1].set_title(f"Correlation Matrix ({df2_name})")

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

    print("-" * 50)  # Separator

    # Find numerical columns common to both DataFrames
    df1_numerical = df1.select_dtypes(include=np.number)
    df2_numerical = df2.select_dtypes(include=np.number)

    numerical_cols1 = set(df1_numerical.columns)
    numerical_cols2 = set(df2_numerical.columns)
    common_numerical_cols = list(numerical_cols1.intersection(numerical_cols2))
    common_numerical_cols.sort()  # Sort for consistent order

    if not common_numerical_cols:
        print("No common numerical columns found to plot distributions side-by-side.")
        return  # Exit the function if no common columns

    print(
        f"Plotting distributions sequentially for common numerical columns: {common_numerical_cols}"
    )

    # --- Plot Distributions of Each Common Numerical Column Side-by-Side (Sequentially) ---

    for i, col in enumerate(common_numerical_cols):
        # Create a NEW figure with two subplots side-by-side for the current column
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 2 columns

        # Plot distribution for df1 on the left subplot
        sns.histplot(df1[col], kde=True, ax=axes[0])
        axes[0].set_title(f"Distribution of {col} ({df1_name})")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Frequency")

        # Plot distribution for df2 on the right subplot
        sns.histplot(df2[col], kde=True, ax=axes[1])
        axes[1].set_title(f"Distribution of {col} ({df2_name})")
        axes[1].set_xlabel(col)
        axes[1].set_ylabel("Frequency")

        plt.tight_layout()  # Adjust layout for THIS figure
        plt.show()  # Display THIS figure immediately

        # Optional: Add a small separator if desired between plots
        # print("-" * 30)


def do_violinplot(
    ds: pd.DataFrame,
    x: str,
    y: str,
    main_title: str = None,
    subtitle: str = None,
    facet_order=None,
):
    """
    ds          : DataFrame containing at least [x, y, 'data']
    x           : column name for the grouping (bool, 'True'/'False', or 0/1)
    y           : name of the numeric column to plot on the y-axis
    main_title  : big title (e.g. "Lung Cancer")
    subtitle    : subtitle (e.g. "Smoking Duration")
    facet_order : list of facet names in the order you want (e.g. ['synthetic','training'])
    """
    # 1) prepare a working copy
    df = ds.copy()

    # 2) normalize x to string then map to Control/Case
    def map_to_label(v):
        if v in (1, "1", True, "True"):
            return "Case"
        elif v in (0, "0", False, "False"):
            return "Control"
        else:
            raise ValueError(f"do_violinplot: unexpected x value {v!r}")

    df["caco_label"] = df[x].apply(map_to_label)

    # 3) define palette
    palette = {"Control": "#0081AF", "Case": "#FF420E"}

    # 4) Set up Seaborn style
    sns.set_style("whitegrid")
    plt.rc("axes.spines", top=False, right=False)

    # 5) Build the FacetGrid
    g = sns.FacetGrid(
        df,
        col="data",
        col_order=facet_order,
        sharey=True,
        despine=False,
        height=8,  # controls single-facet height
        aspect=0.75,  # controls width = height*aspect
        margin_titles=True,
    )

    # 6) Combined violin + dots
    def _violin_with_dots(data, **kwargs):
        ax = plt.gca()
        sns.violinplot(
            data=data,
            x="caco_label",
            y=y,
            palette=palette,
            cut=0,
            inner=None,
            linewidth=1.2,
            zorder=1,
            ax=ax,
        )
        sns.stripplot(
            data=data,
            x="caco_label",
            y=y,
            color="black",
            edgecolor="none",
            size=3,
            alpha=0.3,
            jitter=0.15,
            zorder=2,
            ax=ax,
        )

    g.map_dataframe(_violin_with_dots)

    # 7) Tidy up axes and borders
    for ax in g.axes.flat:
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
        # light gray border
        for spine in ["top", "bottom", "left", "right"]:
            ax.spines[spine].set_color("#CCCCCC")
            ax.spines[spine].set_linewidth(1)

    # 8) Titles
    g.figure.text(0.5, 0.925, f"{main_title} -vs- {subtitle}", ha="center", fontsize=16)

    # 9) Legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=palette["Control"],
            label="Control",
            markersize=8,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=palette["Case"],
            label="Case",
            markersize=8,
        ),
    ]
    g.figure.legend(
        handles=handles,
        title="Case / Control",
        loc="upper right",
        bbox_to_anchor=(0.98, 0.95),
    )

    # 10) Axis label
    g.set_ylabels(y.replace("_", " ").title())

    # 11) Resize the overall figure
    g.figure.set_size_inches(10, 8)  # width=10in, height=24in

    # 12) Layout
    g.figure.tight_layout()
    g.figure.subplots_adjust(top=0.88)  # room for title


def analyze_dfs(df1, df2, df3, df4, list_columns, list_discrete_columns=[]):
    # Step 1: Extract selected columns
    dfs = [df[list_columns] for df in [df1, df2, df3, df4]]

    # Step 2: Compute correlations
    corrs = [df.corr() for df in dfs]

    # Step 3: Compute differences
    diff_1_2 = corrs[0] - corrs[1]
    diff_3_4 = corrs[2] - corrs[3]

    # Step 4: Plot all correlation matrices and differences
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))

    sns.heatmap(
        corrs[0],
        ax=axes[0, 0],
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
    )
    axes[0, 0].set_title("Correlation Matrix: df_real")

    sns.heatmap(
        corrs[1],
        ax=axes[0, 1],
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
    )
    axes[0, 1].set_title("Correlation Matrix: df_fake")

    # Plot the two differences
    sns.heatmap(
        diff_1_2,
        ax=axes[0, 2],
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
    )
    axes[0, 2].set_title("Correlation Difference: df_real - df_fake")

    sns.heatmap(
        corrs[2],
        ax=axes[1, 0],
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
    )
    axes[1, 0].set_title("Correlation Matrix: DF3")

    sns.heatmap(
        corrs[3],
        ax=axes[1, 1],
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
    )
    axes[1, 1].set_title("Correlation Matrix: DF4")

    sns.heatmap(
        diff_3_4,
        ax=axes[1, 2],
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
    )
    axes[1, 2].set_title("Correlation Difference: DF3 - DF4")

    plt.tight_layout()
    plt.show()

    # Step 5: Plot distributions for each column
    for col in list_columns:
        col_dtype = df1[col].dtype

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        if (
            pd.api.types.is_numeric_dtype(col_dtype)
            and col not in list_discrete_columns
        ):
            # Plot KDEs for numeric columns
            sns.kdeplot(
                df1[col].dropna(),
                ax=axes[0],
                label="Real",
                color="blue",
                fill=True,
                alpha=0.4,
            )
            sns.kdeplot(
                df2[col].dropna(),
                ax=axes[0],
                label="Synthetic",
                color="orange",
                fill=True,
                alpha=0.4,
            )
            axes[0].set_title(f"KDE Plot of {col}: df_real vs df_fake")
            axes[0].legend()

            sns.kdeplot(
                df3[col].dropna(),
                ax=axes[1],
                label="DF3",
                color="blue",
                fill=True,
                alpha=0.4,
            )
            sns.kdeplot(
                df4[col].dropna(),
                ax=axes[1],
                label="DF4",
                color="orange",
                fill=True,
                alpha=0.4,
            )
            axes[1].set_title(f"KDE Plot of {col}: DF3 vs DF4")
            axes[1].legend()

        elif pd.api.types.is_object_dtype(
            col_dtype
        ) or pd.api.types.is_categorical_dtype(col_dtype):
            # Plot bar plots for categorical columns
            df1_counts = df1[col].value_counts(normalize=True)
            df2_counts = df2[col].value_counts(normalize=True)
            df3_counts = df3[col].value_counts(normalize=True)
            df4_counts = df4[col].value_counts(normalize=True)

            df12 = pd.DataFrame({"df_real": df1_counts, "df_fake": df2_counts}).fillna(
                0
            )
            df34 = pd.DataFrame({"DF3": df3_counts, "DF4": df4_counts}).fillna(0)

            df12.plot(kind="bar", ax=axes[0], alpha=0.7)
            axes[0].set_title(f"Category Plot of {col}: df_real vs df_fake")
            axes[0].set_ylabel("Proportion")

            df34.plot(kind="bar", ax=axes[1], alpha=0.7)
            axes[1].set_title(f"Category Plot of {col}: DF3 vs DF4")
            axes[1].set_ylabel("Proportion")

        else:
            print(f"Skipping column {col} due to unsupported data type: {col_dtype}")
            plt.close(fig)
            continue

        plt.tight_layout()
        plt.show()


def analyze_fake_vs_real_single_run_pdf(
    df_real,
    df_fake,
    list_columns=None,
    list_discrete_columns=[],
    output_pdf=None,
    violin_specs: list[tuple] = [],
):
    """
    Compare df1 vs. df2 on selected columns and (optionally) append violin plots.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        The two dataframes to compare.
    list_columns : list[str], optional
        Columns to include in correlation + distribution comparison.
        Defaults to all columns of df1.
    list_discrete_columns : list[str], optional
        Columns to treat as discrete even if numeric.
    output_pdf : str | None
        Path to a PDF file to save all figures. If None, figures are shown interactively.
    violin_specs : list of tuples (y, main_title, subtitle), optional
        After the correlation + KDE/bar pages, for each tuple:
          - y: column name to plot on the y-axis
          - main_title: big title for that violin page
          - subtitle: subtitle for that violin page
    """
    pdf = PdfPages(output_pdf) if output_pdf else None

    # --- Correlation comparison ---
    if list_columns is None:
        list_columns = list(df_real.columns)

    sub1 = df_real[list_columns]
    sub2 = df_fake[list_columns]

    corr1 = sub1.corr()
    corr2 = sub2.corr()
    diff_corr = corr1 - corr2

    fig, axes = plt.subplots(3, 1, figsize=(10, 24))  # changed from 1 row to 3 rows
    sns.heatmap(corr1, ax=axes[0], cmap="coolwarm", center=0, annot=True, fmt=".2f")
    axes[0].set_title("Correlation Matrix: df_real")
    sns.heatmap(corr2, ax=axes[1], cmap="coolwarm", center=0, annot=True, fmt=".2f")
    axes[1].set_title("Correlation Matrix: df_fake")
    sns.heatmap(diff_corr, ax=axes[2], cmap="coolwarm", center=0, annot=True, fmt=".2f")
    axes[2].set_title("Correlation Difference: df_real - df_fake")
    plt.tight_layout()

    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    else:
        plt.show()

    # --- Distribution comparison ---
    num_cols = len(list_columns)
    for i, col in enumerate(list_columns):
        # if (
        #     "date" in col
        #     or "rand36" in col
        #     or "qol" in col
        #     or "issi" in col
        #     or "cage" in col
        # ):
        #     continue
        print(f">> processing {i+1}/{num_cols}: {col}")

        col_dtype = df_real[col].dtype
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        # Calculate percentage of missing values BEFORE any modifications like dropna() or astype(str)
        # Ensure division by zero is handled if a dataframe is empty (though unlikely for a column)
        missing_real_pct = (
            (df_real[col].isnull().sum() / len(df_real[col]) * 100)
            if len(df_real[col]) > 0
            else 0
        )
        missing_fake_pct = (
            (df_fake[col].isnull().sum() / len(df_fake[col]) * 100)
            if len(df_fake[col]) > 0
            else 0
        )
        title_suffix = f"\nMissing: df_real {missing_real_pct:.2f}%, df_fake {missing_fake_pct:.2f}%"

        if (
            pd.api.types.is_numeric_dtype(col_dtype)
            and col not in list_discrete_columns
        ):
            sns.kdeplot(
                df_real[col].dropna(),
                label="Real",
                ax=ax,
                color="blue",
                fill=True,
                alpha=0.4,
            )
            sns.kdeplot(
                df_fake[col].dropna(),
                label="Synthetic",
                ax=ax,
                color="orange",
                fill=True,
                alpha=0.4,
            )
            ax.set_title(f"KDE Plot of {col}: df_real vs df_fake" + title_suffix)
            ax.legend()

        else:
            df_real[col] = df_real[col].astype(str)
            df_fake[col] = df_fake[col].astype(str)

            counts1 = df_real[col].value_counts(normalize=True)
            counts2 = df_fake[col].value_counts(normalize=True)
            df_cat = pd.DataFrame({"df_real": counts1, "df_fake": counts2}).fillna(0)
            df_cat.plot(kind="bar", ax=ax, alpha=0.7)
            ax.set_title(f"Category Plot of {col}: df_real vs df_fake" + title_suffix)
            ax.set_ylabel("Proportion")

        plt.tight_layout()
        if pdf:
            pdf.savefig(fig)
            plt.close(fig)
        else:
            plt.show()

    # --- Optional violin pages ---
    if violin_specs:
        for x, y, main_title, subtitle in violin_specs:
            # This function calls plt.show() internally, so we'll suppress that
            fig = do_violinplot(
                ds=pd.concat(
                    [df_real.assign(data="df_real"), df_fake.assign(data="df_fake")]
                ),
                x=x,
                y=y,
                main_title=main_title,
                subtitle=subtitle,
                facet_order=["df_real", "df_fake"],
            )
            if pdf:
                pdf.savefig(fig)
                plt.close(fig)
            else:
                plt.show()

    if pdf:
        pdf.close()


def analyze_fake_vs_real_separate_pdf(
    df_real,
    df_fake,
    list_columns=None,
    list_discrete_columns=[],
    output_pdf=None,
    violin_specs: list[tuple] = [],
    fig_width=9,
    mode="preprocessing",
):
    """
    Compare df_real vs. df_fake on selected columns and (optionally) append violin plots.
    Each figure is saved into a separate PDF file.

    Parameters
    ----------
    df_real, df_fake : pd.DataFrame
        The two dataframes to compare.
    list_columns : list[str], optional
        Columns to include in correlation + distribution comparison.
        Defaults to all columns of df_real.
    list_discrete_columns : list[str], optional
        Columns to treat as discrete even if numeric.
    output_pdf : str | None
        Path prefix for saving PDF files. If None, figures are shown interactively.
        Example: output_pdf="results/analysis" → saves as "results/analysis_correlation.pdf", etc.
    violin_specs : list of tuples (x, y, main_title, subtitle), optional
        For each tuple, creates a violin plot page.
    """

    custom_red = config.COLORS["red"]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_green", ["white", custom_red]
    )

    colors = [
        adjust_color("gray", 100),  # Real data (gray)
        adjust_color("red", 100),
        adjust_color("blue", 100),
        adjust_color("orange", 100),
        adjust_color("green", 100),
    ]

    linestyles = [
        "solid",
        "solid",
        "solid",
        "solid",
        "solid",
    ]

    linewidths = [1, 2, 2, 2, 2]

    # make sure we have a safe prefix for saving
    prefix = output_pdf if output_pdf else None

    if list_columns is None:
        list_columns = list(df_real.columns)

    # --- Correlation comparison ---
    sub1 = df_real[list_columns]
    sub2 = df_fake[list_columns]

    corr1 = sub1.corr()
    corr2 = sub2.corr()
    diff_corr = abs(corr1 - corr2)

    if mode == "preprocessing":
        fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Define a diverging colormap with a range from 0 to 1 for the heatmap
    # This creates a blue-white-red colormap where 0 is white, negative values are blue, positive are red.
    # Since the diff_corr can be negative, a diverging colormap is suitable.
    # The 'center=0' parameter ensures that 0 is the neutral point in the colormap.
    heatmap = sns.heatmap(
        diff_corr,
        cmap="RdBu_r",  # Use a diverging colormap (Red-Blue, reversed so red is positive difference)
        center=0,  # Center the colormap at 0 difference
        annot=False,  # Set annot to False to remove numerical annotations
        fmt=".2f",
        vmin=0,  # Set the minimum value for the colormap range
        vmax=2,  # Set the maximum value for the colormap range
    )
    # Access the colorbar and set the font size of tick labels
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(
        labelsize=XYTICKS_PARAMS_FONTSIZE
    )  # Set font size of colorbar tick labels to 12

    if mode == "preprocessing":
        plt.xticks(
            rotation=45, ha="right", fontsize=XYTICKS_PARAMS_FONTSIZE
        )  # Set x-tick font size
        plt.yticks(
            rotation=0, ha="right", fontsize=XYTICKS_PARAMS_FONTSIZE
        )  # Set y-tick font size
    else:
        plt.xticks([])
        plt.yticks([])
    # ax.set_title(
    #     "Correlation Difference", fontsize=TITLE_FONTSIZE
    # )  # Set title font size

    plt.tight_layout()

    if prefix:
        fig.savefig(f"{prefix}_correlation.pdf")
        plt.close(fig)
    else:
        plt.show()

    # --- Distribution comparison ---
    num_cols = len(list_columns)
    for i, col in enumerate(list_columns):
        print(f">> processing {i+1}/{num_cols}: {col}")

        col_dtype = df_real[col].dtype

        if (
            pd.api.types.is_numeric_dtype(col_dtype)
            and col not in list_discrete_columns
        ):
            if mode == "preprocessing":
                fig, ax = plt.subplots(1, 1, figsize=(9, 5))
            else:
                fig, ax = plt.subplots(1, 1, figsize=(6, 4))

            sns.kdeplot(
                df_real[col].dropna(),
                label="Real",
                color=colors[0],  # Gray color for real data
                linestyle=linestyles[0],  # Solid linestyle
                linewidth=linewidths[0],  # Thicker linewidth
                bw_adjust=0.8,
                ax=ax,
                fill=True,
                zorder=2,  # Bring to front using zorder
            )
            sns.kdeplot(
                df_fake[col].dropna(),
                label="Synthetic",
                color=colors[1],  # Gray color for real data
                linestyle=linestyles[1],  # Solid linestyle
                linewidth=linewidths[1],  # Thicker linewidth
                bw_adjust=0.8,
                ax=ax,
                # fill=True,
                zorder=1,
                # alpha=0.4,
            )
            # ax.set_title(
            #     f"{col}",
            #     fontsize=TITLE_FONTSIZE,
            # )

            plt.xticks(
                rotation=0, ha="right", fontsize=XYTICKS_PARAMS_FONTSIZE
            )  # Set x-tick font size
            plt.yticks(
                rotation=0, ha="right", fontsize=XYTICKS_PARAMS_FONTSIZE
            )  # Set y-tick font size

            ax.legend()
            ax.set_xlabel("")  # Remove x-axis title
            ax.set_ylabel("")  # Remove y-axis title
        else:
            if mode == "preprocessing":
                fig, ax = plt.subplots(1, 1, figsize=(9, 6))
            else:
                fig, ax = plt.subplots(1, 1, figsize=(6, 5))

            # --- Define the replacement mappings for categories ---
            # You can customize these short names as needed
            health_map = {
                "fairly poor": "f. poor",
                "poor": "poor",
                "moderate": "moderate",
                "good": "good",
                "very good": "v. good",
            }

            education_map = {
                "primary school": "primary",
                "secondary school": "secondary",
                "high school": "high",
                "university": "university",
            }

            df_real[col] = df_real[col].astype(str)
            df_fake[col] = df_fake[col].astype(str)

            # --- Apply replacements for specific columns ---
            if "health" in col:
                df_real[col] = df_real[col].replace(health_map)
                df_fake[col] = df_fake[col].replace(health_map)
            elif "education" in col:
                df_real[col] = df_real[col].replace(education_map)
                df_fake[col] = df_fake[col].replace(education_map)

            counts1 = df_real[col].value_counts(normalize=True)
            counts2 = df_fake[col].value_counts(normalize=True)
            df_cat = pd.DataFrame({"Real": counts1, "Synthetic": counts2}).fillna(0)
            df_cat = df_cat.rename(index={"-1": "missing", "nan": "missing"})

            # --- Sort the DataFrame alphabetically by category names (the index) ---
            df_cat = df_cat.sort_index()

            # --- Plot the bar chart with specified colors ---
            df_cat.plot(
                kind="bar",
                ax=ax,
                alpha=0.7,
                color=[colors[0], colors[1]],  # Set colors for 'Real' and 'Synthetic'
            )
            # ax.set_title(
            #     f"{col}",
            #     fontsize=TITLE_FONTSIZE,
            # )

            plt.xticks(
                rotation=45, ha="right", fontsize=XYTICKS_PARAMS_FONTSIZE
            )  # Set x-tick font size
            plt.yticks(
                rotation=0, ha="right", fontsize=XYTICKS_PARAMS_FONTSIZE
            )  # Set y-tick font size

            ax.set_ylabel("")
            ax.set_xlabel("")  # Remove x-axis title for consistency where possible

        plt.legend(
            fontsize=XYTICKS_PARAMS_FONTSIZE
        )  # You can use a numeric value (e.g., 10, 12, 14)
        plt.tight_layout()
        if prefix:
            safe_col = col.replace("/", "_").replace("\\", "_")
            fig.savefig(f"{prefix}_distribution_{safe_col}_{fig_width}.pdf")
            plt.close(fig)
        else:
            plt.show()

    # --- Optional violin pages ---
    if violin_specs:
        for idx, (x, y, main_title, subtitle) in enumerate(violin_specs, start=1):
            # Assuming do_violinplot is a function defined elsewhere and takes these arguments
            fig = do_violinplot(
                ds=pd.concat(
                    [df_real.assign(data="df_real"), df_fake.assign(data="df_fake")]
                ),
                x=x,
                y=y,
                main_title=main_title,
                subtitle=subtitle,
                facet_order=["df_real", "df_fake"],
            )
            if prefix:
                fig.savefig(f"{prefix}_violin_{idx}_{y}.pdf")
                plt.close(fig)
            else:
                plt.show()

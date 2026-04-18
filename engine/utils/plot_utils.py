import re
import matplotlib.pyplot as plt
from engine.utils.eval_utils import compute_dwp
from engine.config import config


def plot_dwp(df1, df2, title="", is_plot=True):
    d, x, y = compute_dwp(df1, df2)
    if is_plot:
        _x = [0, 1]
        _y = [0, 1]
        plt.plot(_x, _y)
        plt.scatter(x, y, s=2)
        plt.xlabel("real")
        plt.ylabel("fake")
        plt.title(f"{title}: {d:.6f}")
        plt.show()


def plot_legend_bottom(ax):
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
        shadow=True,
        ncol=5,
    )


def plot_legend_right(ax):
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))


def map_biobank_dataset_plot(name):
    """
    Cleans the biobank dataset name by removing prefixes and applying specific renaming rules.

    Args:
        name (str): Original dataset name.

    Returns:
        str: Cleaned name.
    """
    # Remove prefixes
    name = re.sub(r"^biobank_sen_|^biobank_", "", name)

    # Apply specific renaming rules
    if "record_dead" in name:
        return "dead record"
    elif "patient_dead" in name:
        return "dead patient"

    # Remove "record_" if present
    name = name.replace("record_", "")

    # Remove the last part after the last "_"
    if "phase3" in name:
        name = name.replace("phase3_", "")
        return name
    if "_" in name:
        name = "_".join(name.split("_")[:-1])

    return name


def _rename_biobank_datasets(df):
    """
    Renames all columns in a DataFrame using the biobank name cleaning function.

    Args:
        df (pd.DataFrame): Input DataFrame with original column names.

    Returns:
        pd.DataFrame: DataFrame with renamed columns.
    """
    df = df.copy()
    df.columns = [map_biobank_dataset_plot(col) for col in df.columns]

    name_mapping = {}

    # Rename columns in the DataFrame
    df = df.rename(columns=name_mapping)

    return df


def rename_biobank_datasets(df):
    """
    Renames all columns in a DataFrame using the biobank name cleaning function.

    Args:
        df (pd.DataFrame): Input DataFrame with original column names.

    Returns:
        pd.DataFrame: DataFrame with renamed columns.
    """
    orig_to_new = {
        "biobank_record_vital": "L1 Vital",
        "biobank_record_dead": "L1 Dead R",
        "biobank_record_icd7": "L1 ICD7",
        "biobank_record_icd9": "L1 ICD9",
        "biobank_record_icdo2": "L1 ICD-O2",
        "biobank_record_icdo3": "L1 ICD-O3",
        "biobank_patient_dead": "L1 Dead P",
        "biobank_sen_meta": "L2 Meta",
        "biobank_sen_prostate": "L2 Prostate",
        "biobank_sen_breast": "L2 Breast",
        "biobank_sen_colorectal": "L2 Colorectal",
        "biobank_sen_uroandkid": "L2 Uro & Kidney",
        "biobank_sen_lung": "L2 Lung",
        "biobank_sen_pancreatic": "L2 Pancreatic",
        "biobank_sen_haematological": "L2 Haematological",
        "biobank_phase3_cancer_all": "L3 Cancer",
        "biobank_phase3_cancer_bio_all": "L3 Cancer Bio",
        "biobank_phase3_cancer_185": "L3 Cancer 185",
        "biobank_phase3_cancer_174": "L3 Cancer 174",
        "biobank_phase3_cancer_153": "L3 Cancer 153",
        "biobank_phase3_cancer_162": "L3 Cancer 162",
        "biobank_phase3_cancer_188": "L3 Cancer 188",
        "biobank_phase3_cancer_172": "L3 Cancer 172",
        "biobank_phase3_cancer_154": "L3 Cancer 154",
        "biobank_phase3_cancer_173": "L3 Cancer 173",
        "biobank_phase3_cancer_200": "L3 Cancer 200",
    }

    df = df.copy()
    df.columns = [orig_to_new[col] for col in df.columns]

    name_mapping = {}

    # Rename columns in the DataFrame
    df = df.rename(columns=name_mapping)

    return df


def map_biobank_model_plot(model_str, is_full=True):
    """
    Convert <model>_<is_condition_vector>_<loss_version> to ModelType-CV-Loss.

    Args:
        model_str (str): Model name in the format "<model>_<is_condition_vector>_<loss_version>".

    Returns:
        str: Formatted model name in "ModelType-CV-Loss" format.
    """
    # Mapping for loss version
    loss_mapping = {"0": "BaseFn", "2": "CorrDst"}

    # Split model name
    parts = model_str.split("_")
    if len(parts) != 3:
        raise ValueError(
            "Input format must be '<model>_<is_condition_vector>_<loss_version>'"
        )

    model_type, is_cv, loss_version = parts

    cv_str = "" if is_cv.lower() in ["c", "true", "1"] else "*"

    # Get loss version name
    loss_str = loss_mapping.get(loss_version, f"Unknown{loss_version}")

    if "tabddpm" == model_type:
        model_type = config.DICT_MAPPING_MODEL[f"{model_type}-{is_cv}"]
        if not is_full:
            return f"{model_type}"
        else:
            return f"{model_type} + {loss_str}"
    elif "tabsyn" == model_type:
        model_type = config.DICT_MAPPING_MODEL[model_type]
        if not is_full:
            return f"{model_type}"
        else:
            return f"{model_type} + {loss_str}"
    else:
        model_type = config.DICT_MAPPING_MODEL[model_type]
        if not is_full:
            return f"{model_type}"
        else:
            return f"{model_type}{cv_str} + {loss_str}"


def rename_dataframe_models(df):
    """
    Renames the index of a DataFrame using the `format_model_name` function.

    Args:
        df (pd.DataFrame): DataFrame with model names as index.

    Returns:
        pd.DataFrame: DataFrame with renamed index.
    """
    df = df.copy()
    df.index = df.index.map(map_biobank_model_plot)
    return df

import pandas as pd
import json


def export_unique_counts(df, exclude_substrings, json_filename):
    # Initialize dictionary to store the unique counts for each column
    counts_dict = {}

    # Loop over each column in the DataFrame
    for col in df.columns:
        # Check if any substring in exclude_substrings is in the column name
        if not any(substring in col for substring in exclude_substrings):
            # Calculate the frequency (count) of each unique value in the column
            unique_counts = df[col].value_counts(dropna=True).to_dict()
            # Store the frequency dictionary in the main dictionary
            counts_dict[col] = unique_counts

    # Write the dictionary to the specified JSON file
    with open(json_filename, "w") as json_file:
        json.dump(counts_dict, json_file, indent=4)

    print(f"Unique counts written to {json_filename}")


def label_cancer_type_icd9(df):
    df["icd9_code"] = df["icd9_code"].fillna(0).astype(int).astype(str)

    # Define ICD-9 code ranges for each cancer type without periods
    cancer_code_ranges = {
        "Prostate": [f"185{i}" for i in range(10)],
        "Breast": [f"174{i}" for i in range(10)] + [f"175{i}" for i in range(10)],
        "Colorectal": [f"153{i}" for i in range(10)] + [f"154{i}" for i in range(9)],
        "Urothelial and Kidney": [f"188{i}" for i in range(10)] + ["1890", "1891"],
        "Lung": [f"162{i}" for i in range(10)],
        "Pancreatic": [f"157{i}" for i in range(10)],
        "Haematological": [f"204{i}" for i in range(10)]
        + [f"205{i}" for i in range(10)]
        + [f"206{i}" for i in range(10)]
        + [f"207{i}" for i in range(10)]
        + [f"208{i}" for i in range(10)]
        + [f"200{i}" for i in range(10)]
        + [f"201{i}" for i in range(10)]
        + [f"202{i}" for i in range(10)]
        + [f"203{i}" for i in range(9)],
    }

    # Initialize the new column with NaN values
    df["icd9_cancer"] = pd.NA

    # Iterate over each cancer type and update the new column based on code ranges
    for cancer_type, codes in cancer_code_ranges.items():
        df.loc[df["icd9_code"].isin(codes), "icd9_cancer"] = cancer_type

    return df


def label_cancer_type_icd7(df):
    df["icd7_code"] = df["icd7_code"].fillna("0").astype(str)

    # Dictionary with prefixes for each cancer type in ICD-7
    cancer_code_prefixes = {
        "Prostate": "177",
        "Breast": "170",
        "Colorectal": ["153", "154"],
        "Urothelial and Kidney": ["180", "181"],
        "Lung": "162",
        "Pancreatic": "157",
        "Haematological": [
            "200",
            "201",
            "202",
            "203",
            "204",
            "205",
            "206",
            "207",
            "208",
        ],
    }

    # Create a new column for cancer type and initialize it with NaN
    df["icd7_cancer"] = pd.NA

    # Iterate over each cancer type and label rows accordingly
    for cancer_type, prefixes in cancer_code_prefixes.items():
        if isinstance(prefixes, list):
            # Combine multiple prefixes for one type
            pattern = "^(" + "|".join(prefixes) + ")"
        else:
            # Single prefix pattern
            pattern = "^" + prefixes

        # Update the 'icd7_cancer' column based on matching codes
        df.loc[df["icd7_code"].str.match(pattern), "icd7_cancer"] = cancer_type

    return df


def aggregate_events_with_final_fill(
    df,
    C_same=["avanid", "birthdate", "dead_by_april2022", "sex"],
    C_dif=[
        "exclude_diet",
        "inclusion_age",
        "fasting_q",
        "height",
        "weight",
        "bmi",
        "waist",
        "total_cholesterol",
        "hdl",
        "ldl",
        "triglycerides",
        "bloods_0h",
        "bloods_2h",
        "systolic_bp",
        "diastolic_bp",
        "marital_status",
        "education",
        "eventdate",
    ],
):
    # Sort dataframe by 'avanid' and 'eventdate'
    df = df.sort_values(["avanid", "eventdate"])

    # Initialize an empty list to store merged rows
    merged_data = []

    # Group by 'avanid'
    for avanid, group in df.groupby("avanid"):
        # Collect 'C_same' values (assumed to be identical within each avanid group)
        merged_row = {col: group[col].iloc[0] for col in C_same}
        merged_row["avanid"] = avanid

        # Get C_dif columns for events in chronological order
        dif_values = group[C_dif].values
        num_events = len(dif_values)

        # Calculate the start position for filling based on the number of events
        start_pos = 5 - num_events

        # Initialize sets for C_dif with NaN
        for i in range(5):
            if i >= start_pos:
                # Fill with event data for positions starting from the calculated position
                event_data = dif_values[i - start_pos]
                for j, col in enumerate(C_dif):
                    merged_row[f"{col}_{i + 1}"] = event_data[j]
            else:
                # Fill with NaN for earlier sets
                for col in C_dif:
                    merged_row[f"{col}_{i + 1}"] = pd.NA

        # Append merged_row to the list
        merged_data.append(merged_row)

    # Convert merged data to DataFrame
    return pd.DataFrame(merged_data)


def get_info_cancer(df_nshds, df_rcc, df_journal):
    label_cancer_type_icd9(df_nshds)

    export_unique_counts(
        df_nshds,
        exclude_substrings=["avanid", "date", "id_predict"],
        json_filename="df_nshds_unique_values.json",
    )

    export_unique_counts(
        df_rcc,
        exclude_substrings=["avanid", "avan_ridx", "date"],
        json_filename="df_rcc_unique_values.json",
    )

    export_unique_counts(
        df_journal,
        exclude_substrings=["avanid", "ID", "datum"],
        json_filename="df_journal_unique_values.json",
    )


def read_all_tables():
    # read questionaire
    df_nshds = pd.read_csv(
        "minh_predict_dataexport/nshds/gold/nshds_questionaire.tsv",
        sep="\t",
        header=0,
    )
    df_nshds = df_nshds.drop(
        columns=[
            "id_predict",
            "predict_cohort",
            "consent_predict",
            "in_subcohort",
            "questionaire_date",
            "diet_questionaire",
            "consent_nshds",
            "bloodvalues_at_sampledate",
            "cohort",
            "h_w_estimate",
            "after_090901",
            "eventtype",
            "projecttags",
            # "fasting_q", # we add more columns based on this one because one patient can have multiple measurements
        ]
    )
    print(df_nshds)
    print(df_nshds["avanid"].value_counts())

    # read icd
    df_rcc = pd.read_csv(
        "minh_predict_dataexport/rcc/gold/ps_RCC-ICD_register.tsv",
        sep="\t",
        header=0,
    )
    df_rcc = df_rcc.drop(
        columns=[
            "avan_ridx",
            "sex",
            "dead_at_match_date",
            "death_match_date",
            "death_match_date_original",
            "autopsy_discovery",
            "pat_cyt_dep",
            "specimen_year",
            "specimen_number",
            "eventtype",
            "projecttags",
        ],
    )
    print(df_rcc)

    # read patient journals
    df_journal = pd.read_csv(
        "minh_predict_dataexport/icd_rv/silver/ps_dxcode-rv_register.tsv",
        sep="\t",
        header=0,
    )
    df_journal = df_journal[
        [
            "avanid",
            "Diagnosdatum",
            "Diagnoskod",
            "Åtgärdsdatum",
            "Åtgärdskod",
        ]
    ]
    print(df_journal)

    # read metabolomics
    df_meta = pd.read_csv(
        "minh_predict_dataexport/nightingale/bronze/1967905 PREDICT-16-Jan-2024-Results.csv",
        sep=",",
        header=0,
    )
    print(df_meta)

    # read metabolomics info
    df_meta_info = pd.read_csv(
        "minh_predict_dataexport/nightingale_samplesheets/1967905 PREDICT-16-Jan-2024-Results_key.tsv",
        sep="\t",
        header=0,
    )

    print(df_meta_info)

    return df_nshds, df_rcc, df_journal, df_meta, df_meta_info


def merge_and_check_cancer(df):
    # Step 1: Merge 'icd7_cancer' and 'icd9_cancer' into a single 'cancer' column
    df["cancer"] = df["icd7_cancer"].fillna(df["icd9_cancer"])
    df = df.fillna(-1)

    icd7_na_list = df[
        (df["cancer"] == df["icd7_code"])
        & (~df["icd7_code"].isin([-1, "-1.0", "-1", pd.NA]))
    ]["icd7_code"].tolist()

    icd9_na_list = df[
        (df["cancer"] == df["icd9_code"])
        & (~df["icd9_code"].isin([-1, "-1.0", "-1", pd.NA]))
    ]["icd9_code"].tolist()

    check_data = {"icd7_na_list": icd7_na_list, "icd9_na_list": icd9_na_list}

    with open("check.json", "w") as json_file:
        json.dump(check_data, json_file)

    return df


import pandas as pd


def generate_cancer_dfs(df_merge):
    # Define cancer types and corresponding filenames
    cancer_types = {
        "Prostate": "prostate",
        "Breast": "breast",
        "Colorectal": "colorectal",
        "Urothelial and Kidney": "uroandkid",
        "Lung": "lung",
        "Pancreatic": "pancreatic",
        "Haematological": "haematological",
    }

    # Iterate through each cancer type
    for cancer, file_name in cancer_types.items():
        # Filter the dataframe for the current cancer type
        cancer_df = df_merge[df_merge["cancer"] == cancer]

        # Save the filtered dataframe to a CSV file
        cancer_df.to_csv(f"df_{file_name}.csv", index=False)

    print("CSV files for each cancer type have been generated.")


def preprocess(df_nshds, df_rcc, df_journal, df_meta, df_meta_info):
    df_nshds = aggregate_events_with_final_fill(df_nshds)
    df_nshds = df_nshds.fillna(-1)
    print(df_nshds)
    # df_nshds.to_csv("df_nshds.csv", sep="\t", encoding="utf-8")

    df_rcc = df_rcc.fillna(-1)
    print(df_rcc)

    df_journal = df_journal.fillna(-1)
    print(df_journal)

    # merge df_nshds and df_rcc
    df_merge = pd.merge(
        df_nshds, df_rcc, left_on="avanid", right_on="avanid", how="left"
    )

    # merge meta
    df_meta_info = df_meta_info.dropna(subset=["avanid"])
    df_merged_meta = df_meta.merge(
        df_meta_info[["id_nightingale", "avanid"]],
        left_on="Sample id",
        right_on="id_nightingale",
        how="inner",
    )

    # Drop the 'id_nightingale' column from the merged result as it’s not needed
    df_merged_meta = df_merged_meta.drop(
        columns=[
            "id_nightingale",
            "Sample id",
        ],
    )

    # merge all
    df_merge = pd.merge(
        df_merge, df_merged_meta, left_on="avanid", right_on="avanid", how="left"
    )

    df_merge = df_merge.fillna(-1)

    # get cancer info
    print()
    print()
    print("get_number_per_cancer_icd7(df_merge)")
    print()
    print()
    df_merge = label_cancer_type_icd7(df_merge)
    print()
    print()
    print("get_number_per_cancer_icd9(df_merge)")
    print()
    print()
    df_merge = label_cancer_type_icd9(df_merge)
    print()
    print()
    df_merge = merge_and_check_cancer(df_merge)
    print()
    print()

    # drop icd columns
    df_merge = df_merge.drop(
        columns=[
            "avanid",
            "birthdate",  # keep
            "icd7_code",
            "icd7_text",
            "icd9_code",
            "icd9_text",
            "icdo2_code",
            "icdo2_text",
            "icdo3_code",
            "icdo3_text",
            "c24_code",
            "c24_text",
            "snomed2_code",
            "snomed2_text",
            "snomed3_code",
            "snomed3_text",
            "snomed3_blood",
            "death_date",  # keep
            "icd7_cancer",
            "icd9_cancer",
        ]
    )

    # drop first measure measurements most nan
    df_merge = df_merge.drop(
        columns=[
            "exclude_diet_1",
            "inclusion_age_1",
            "fasting_q_1",
            "height_1",
            "weight_1",
            "bmi_1",
            "waist_1",
            "total_cholesterol_1",
            "hdl_1",
            "ldl_1",
            "triglycerides_1",
            "bloods_0h_1",
            "bloods_2h_1",
            "systolic_bp_1",
            "diastolic_bp_1",
            "marital_status_1",
            "education_1",
            "eventdate_1",
        ]
    )

    # Function to strip whitespace if the element is a string and lowercase
    df_merge = df_merge.applymap(
        lambda x: x.strip().lower() if isinstance(x, str) else x
    )

    # write
    print(df_merge)
    df_merge.to_csv("df_merge.csv", sep="\t", encoding="utf-8")

    return df_merge


def main():
    df_nshds, df_rcc, df_journal, df_meta, df_meta_info = read_all_tables()
    df_merge = preprocess(df_nshds, df_rcc, df_journal, df_meta, df_meta_info)

    # df_merge.to_csv("df_merge.csv", sep="\t", encoding="utf-8")
    print(df_merge)

    # print()
    # print()

    df_merge = pd.read_csv(
        "df_merge.csv",
        sep="\t",
        header=0,
        index_col=0,
    )
    print(df_merge)

    # df_merge = df_merge.drop(
    #     columns=[
    #         # "avanid",
    #         "birthdate",
    #         "icd7_code",
    #         "icd7_text",
    #         "icd9_code",
    #         "icd9_text",
    #         "icdo2_code",
    #         "icdo2_text",
    #         "icdo3_code",
    #         "icdo3_text",
    #         "c24_code",
    #         "c24_text",
    #         "snomed2_code",
    #         "snomed2_text",
    #         "snomed3_code",
    #         "snomed3_text",
    #         "snomed3_blood",
    #         "death_date",
    #         "icd7_cancer",
    #         "icd9_cancer",
    #     ]
    # )

    # Example usage
    # generate_cancer_dfs(df_merge)


if __name__ == "__main__":
    main()


"""
>>> pd.unique(df_nshds["avanid"]).shape
(50087,)

>>> pd.unique(df_journal["avanid"]).shape
(32996,)

>>> pd.unique(df_rcc["avanid"]).shape
(14698,)

>>> pd.unique(df_meta_info["avanid"]).shape
(3284,)

>>> pd.unique(df_journal["Diagnoskod"]).shape
(2687,)

>>> pd.unique(df_journal["Åtgärdskod"]).shape
(3995,)

>>> # Group by 'Diagnoskod' and 'Åtgärdskod', then count unique 'avanid' in each group
>>> unique_counts = df_journal.groupby(['Diagnoskod', 'Åtgärdskod'])['avanid'].nunique()
>>> 
>>> # Print the result
>>> print(unique_counts)
Diagnoskod  Åtgärdskod
A392        DG015         1
            TKC20         1
A398        GD001         1
            XV017         1
A400        0000          6
                         ..
S729        DR029         1
            DV071         1
            XS100         9
            XV017         1
S7290       XS100         3
Name: avanid, Length: 107994, dtype: int64
>>> # Group by Diagnoskod and Åtgärdskod
>>> unique_groups_count = df_journal.groupby(['Diagnoskod', 'Åtgärdskod']).ngroups
>>> 
>>> # Print the number of unique groups
>>> print("Number of unique groups:", unique_groups_count)
Number of unique groups: 107994


[
drop -- 'avanid', "birthdate", 'icd7_code', 'icd7_text', 'icd9_code', 'icd9_text', 'icdo2_code', 'icdo2_text', 'icdo3_code', 'icdo3_text', 'c24_code', 'c24_text', 'snomed2_code', 'snomed2_text', 'snomed3_code', 'snomed3_text', 'snomed3_blood', 'death_date', 'icd7_cancer', 'icd9_cancer', 
    
'dead_by_april2022', 'sex', 'exclude_diet_1', 'inclusion_age_1', 'fasting_q_1', 'height_1', 'weight_1', 'bmi_1', 'waist_1', 'total_cholesterol_1', 'hdl_1', 'ldl_1', 'triglycerides_1', 'bloods_0h_1', 'bloods_2h_1', 'systolic_bp_1', 'diastolic_bp_1', 'marital_status_1', 'education_1', 'eventdate_1', 'exclude_diet_2', 'inclusion_age_2', 'fasting_q_2', 'height_2', 'weight_2', 'bmi_2', 'waist_2', 'total_cholesterol_2', 'hdl_2', 'ldl_2', 'triglycerides_2', 'bloods_0h_2', 'bloods_2h_2', 'systolic_bp_2', 'diastolic_bp_2', 'marital_status_2', 'education_2', 'eventdate_2', 'exclude_diet_3', 'inclusion_age_3', 'fasting_q_3', 'height_3', 'weight_3', 'bmi_3', 'waist_3', 'total_cholesterol_3', 'hdl_3', 'ldl_3', 'triglycerides_3', 'bloods_0h_3', 'bloods_2h_3', 'systolic_bp_3', 'diastolic_bp_3', 'marital_status_3', 'education_3', 'eventdate_3', 'exclude_diet_4', 'inclusion_age_4', 'fasting_q_4', 'height_4', 'weight_4', 'bmi_4', 'waist_4', 'total_cholesterol_4', 'hdl_4', 'ldl_4', 'triglycerides_4', 'bloods_0h_4', 'bloods_2h_4', 'systolic_bp_4', 'diastolic_bp_4', 'marital_status_4', 'education_4', 'eventdate_4', 'exclude_diet_5', 'inclusion_age_5', 'fasting_q_5', 'height_5', 'weight_5', 'bmi_5', 'waist_5', 'total_cholesterol_5', 'hdl_5', 'ldl_5', 'triglycerides_5', 'bloods_0h_5', 'bloods_2h_5', 'systolic_bp_5', 'diastolic_bp_5', 'marital_status_5', 'education_5', 'eventdate_5', 'age', 'diagnosis_date', 'diagnosis_place_of_residence', 'lkf_value', 'malignancy_status', 'figo_value', 't_value', 'm_value', 'n_value', 'eventdate', 'EDTA_plasma', 'Citrate_plasma', 'Low_ethanol', 'Medium_ethanol', 'High_ethanol', 'Isopropyl_alcohol', 'N_methyl_2_pyrrolidone', 'Polysaccharides', 'Aminocaproic_acid', 'Low_glucose', 'High_lactate', 'High_pyruvate', 'Low_glutamine_or_high_glutamate', 'Gluconolactone', 'Low_protein', 'Unexpected_amino_acid_signals', 'Unidentified_macromolecules', 'Unidentified_small_molecule_a', 'Unidentified_small_molecule_b', 'Unidentified_small_molecule_c', 'Below_limit_of_quantification', 'Total_C', 'non_HDL_C', 'Remnant_C', 'VLDL_C', 'Clinical_LDL_C', 'LDL_C', 'HDL_C', 'Total_TG', 'VLDL_TG', 'LDL_TG', 'HDL_TG', 'Total_PL', 'VLDL_PL', 'LDL_PL', 'HDL_PL', 'Total_CE', 'VLDL_CE', 'LDL_CE', 'HDL_CE', 'Total_FC', 'VLDL_FC', 'LDL_FC', 'HDL_FC', 'Total_L', 'VLDL_L', 'LDL_L', 'HDL_L', 'Total_P', 'VLDL_P', 'LDL_P', 'HDL_P', 'VLDL_size', 'LDL_size', 'HDL_size', 'Phosphoglyc', 'TG_by_PG', 'Cholines', 'Phosphatidylc', 'Sphingomyelins', 'ApoB', 'ApoA1', 'ApoB_by_ApoA1', 'Total_FA', 'Unsaturation', 'Omega_3', 'Omega_6', 'PUFA', 'MUFA', 'SFA', 'LA', 'DHA', 'Omega_3_pct', 'Omega_6_pct', 'PUFA_pct', 'MUFA_pct', 'SFA_pct', 'LA_pct', 'DHA_pct', 'PUFA_by_MUFA', 'Omega_6_by_Omega_3', 'Ala', 'Gln', 'Gly', 'His', 'Total_BCAA', 'Ile', 'Leu', 'Val', 'Phe', 'Tyr', 'Glucose', 'Lactate', 'Pyruvate', 'Citrate', 'bOHbutyrate', 'Acetate', 'Acetoacetate', 'Acetone', 'Creatinine', 'Albumin', 'GlycA', 'XXL_VLDL_P', 'XXL_VLDL_L', 'XXL_VLDL_PL', 'XXL_VLDL_C', 'XXL_VLDL_CE', 'XXL_VLDL_FC', 'XXL_VLDL_TG', 'XL_VLDL_P', 'XL_VLDL_L', 'XL_VLDL_PL', 'XL_VLDL_C', 'XL_VLDL_CE', 'XL_VLDL_FC', 'XL_VLDL_TG', 'L_VLDL_P', 'L_VLDL_L', 'L_VLDL_PL', 'L_VLDL_C', 'L_VLDL_CE', 'L_VLDL_FC', 'L_VLDL_TG', 'M_VLDL_P', 'M_VLDL_L', 'M_VLDL_PL', 'M_VLDL_C', 'M_VLDL_CE', 'M_VLDL_FC', 'M_VLDL_TG', 'S_VLDL_P', 'S_VLDL_L', 'S_VLDL_PL', 'S_VLDL_C', 'S_VLDL_CE', 'S_VLDL_FC', 'S_VLDL_TG', 'XS_VLDL_P', 'XS_VLDL_L', 'XS_VLDL_PL', 'XS_VLDL_C', 'XS_VLDL_CE', 'XS_VLDL_FC', 'XS_VLDL_TG', 'IDL_P', 'IDL_L', 'IDL_PL', 'IDL_C', 'IDL_CE', 'IDL_FC', 'IDL_TG', 'L_LDL_P', 'L_LDL_L', 'L_LDL_PL', 'L_LDL_C', 'L_LDL_CE', 'L_LDL_FC', 'L_LDL_TG', 'M_LDL_P', 'M_LDL_L', 'M_LDL_PL', 'M_LDL_C', 'M_LDL_CE', 'M_LDL_FC', 'M_LDL_TG', 'S_LDL_P', 'S_LDL_L', 'S_LDL_PL', 'S_LDL_C', 'S_LDL_CE', 'S_LDL_FC', 'S_LDL_TG', 'XL_HDL_P', 'XL_HDL_L', 'XL_HDL_PL', 'XL_HDL_C', 'XL_HDL_CE', 'XL_HDL_FC', 'XL_HDL_TG', 'L_HDL_P', 'L_HDL_L', 'L_HDL_PL', 'L_HDL_C', 'L_HDL_CE', 'L_HDL_FC', 'L_HDL_TG', 'M_HDL_P', 'M_HDL_L', 'M_HDL_PL', 'M_HDL_C', 'M_HDL_CE', 'M_HDL_FC', 'M_HDL_TG', 'S_HDL_P', 'S_HDL_L', 'S_HDL_PL', 'S_HDL_C', 'S_HDL_CE', 'S_HDL_FC', 'S_HDL_TG', 'XXL_VLDL_PL_pct', 'XXL_VLDL_C_pct', 'XXL_VLDL_CE_pct', 'XXL_VLDL_FC_pct', 'XXL_VLDL_TG_pct', 'XL_VLDL_PL_pct', 'XL_VLDL_C_pct', 'XL_VLDL_CE_pct', 'XL_VLDL_FC_pct', 'XL_VLDL_TG_pct', 'L_VLDL_PL_pct', 'L_VLDL_C_pct', 'L_VLDL_CE_pct', 'L_VLDL_FC_pct', 'L_VLDL_TG_pct', 'M_VLDL_PL_pct', 'M_VLDL_C_pct', 'M_VLDL_CE_pct', 'M_VLDL_FC_pct', 'M_VLDL_TG_pct', 'S_VLDL_PL_pct', 'S_VLDL_C_pct', 'S_VLDL_CE_pct', 'S_VLDL_FC_pct', 'S_VLDL_TG_pct', 'XS_VLDL_PL_pct', 'XS_VLDL_C_pct', 'XS_VLDL_CE_pct', 'XS_VLDL_FC_pct', 'XS_VLDL_TG_pct', 'IDL_PL_pct', 'IDL_C_pct', 'IDL_CE_pct', 'IDL_FC_pct', 'IDL_TG_pct', 'L_LDL_PL_pct', 'L_LDL_C_pct', 'L_LDL_CE_pct', 'L_LDL_FC_pct', 'L_LDL_TG_pct', 'M_LDL_PL_pct', 'M_LDL_C_pct', 'M_LDL_CE_pct', 'M_LDL_FC_pct', 'M_LDL_TG_pct', 'S_LDL_PL_pct', 'S_LDL_C_pct', 'S_LDL_CE_pct', 'S_LDL_FC_pct', 'S_LDL_TG_pct', 'XL_HDL_PL_pct', 'XL_HDL_C_pct', 'XL_HDL_CE_pct', 'XL_HDL_FC_pct', 'XL_HDL_TG_pct', 'L_HDL_PL_pct', 'L_HDL_C_pct', 'L_HDL_CE_pct', 'L_HDL_FC_pct', 'L_HDL_TG_pct', 'M_HDL_PL_pct', 'M_HDL_C_pct', 'M_HDL_CE_pct', 'M_HDL_FC_pct', 'M_HDL_TG_pct', 'S_HDL_PL_pct', 'S_HDL_C_pct', 'S_HDL_CE_pct', 'S_HDL_FC_pct', 'S_HDL_TG_pct', 'cancer']



pd.read_csv("database/dataset/biobank_sensitive/df_merge.csv" , sep="\t", header=0, index_col=0)
"""


"""
percent missing values:


dead_by_april2022: 0.00%
sex: 0.00%

exclude_diet_2: 99.92%
inclusion_age_2: 99.51%
fasting_q_2: 99.51%
height_2: 99.54%
weight_2: 99.54%
bmi_2: 99.54%
waist_2: 100.00%
total_cholesterol_2: 99.54%
hdl_2: 99.82%
ldl_2: 100.00%
triglycerides_2: 99.71%
bloods_0h_2: 99.55%
bloods_2h_2: 99.59%
systolic_bp_2: 99.54%
diastolic_bp_2: 99.54%
marital_status_2: 99.51%
education_2: 99.51%
eventdate_2: 99.51%
exclude_diet_3: 84.03%
inclusion_age_3: 81.01%
fasting_q_3: 81.01%
height_3: 81.38%
weight_3: 81.40%
bmi_3: 81.41%
waist_3: 99.99%
total_cholesterol_3: 81.51%
hdl_3: 95.68%
ldl_3: 99.99%
triglycerides_3: 85.63%
bloods_0h_3: 81.45%
bloods_2h_3: 82.25%
systolic_bp_3: 81.56%
diastolic_bp_3: 81.57%
marital_status_3: 81.01%
education_3: 81.01%
eventdate_3: 81.01%
exclude_diet_4: 45.80%
inclusion_age_4: 41.85%
fasting_q_4: 41.85%
height_4: 42.36%
weight_4: 42.40%
bmi_4: 42.42%
waist_4: 82.16%
total_cholesterol_4: 42.62%
hdl_4: 82.80%
ldl_4: 92.25%
triglycerides_4: 48.36%
bloods_0h_4: 42.47%
bloods_2h_4: 44.20%
systolic_bp_4: 42.59%
diastolic_bp_4: 42.63%
marital_status_4: 41.85%
education_4: 41.85%
eventdate_4: 41.85%

exclude_diet_5: 6.11%
inclusion_age_5: 0.00%
fasting_q_5: 0.00%
height_5: 0.48%
weight_5: 0.52%
bmi_5: 0.54%
waist_5: 41.40%
total_cholesterol_5: 0.75%
hdl_5: 41.69%
ldl_5: 54.00%
triglycerides_5: 6.83%
bloods_0h_5: 0.71%
bloods_2h_5: 6.18%
systolic_bp_5: 0.79%
diastolic_bp_5: 0.82%
marital_status_5: 0.00%
education_5: 0.00%
eventdate_5: 0.00%
age: 64.50%
diagnosis_date: 64.50%
diagnosis_place_of_residence: 64.50%
lkf_value: 64.50%
malignancy_status: 64.50%
figo_value: 99.04%
t_value: 81.75%
m_value: 81.84%
n_value: 81.82%
eventdate: 64.50%

metabolamics

EDTA_plasma: 91.33%
Citrate_plasma: 91.33%
Low_ethanol: 91.33%
Medium_ethanol: 91.33%
High_ethanol: 91.33%
Isopropyl_alcohol: 91.33%
N_methyl_2_pyrrolidone: 91.33%
Polysaccharides: 91.33%
Aminocaproic_acid: 91.33%
Low_glucose: 91.33%
High_lactate: 91.33%
High_pyruvate: 91.33%
Low_glutamine_or_high_glutamate: 91.33%
Gluconolactone: 91.33%
Low_protein: 91.33%
Unexpected_amino_acid_signals: 91.33%
Unidentified_macromolecules: 91.33%
Unidentified_small_molecule_a: 91.33%
Unidentified_small_molecule_b: 91.33%
Unidentified_small_molecule_c: 91.33%
Below_limit_of_quantification: 91.33%
Total_C: 91.72%
non_HDL_C: 91.72%
Remnant_C: 91.72%
VLDL_C: 91.72%
Clinical_LDL_C: 91.72%
LDL_C: 91.72%
HDL_C: 91.72%
Total_TG: 91.72%
VLDL_TG: 91.72%
LDL_TG: 91.72%
HDL_TG: 91.72%
Total_PL: 91.72%
VLDL_PL: 91.72%
LDL_PL: 91.72%
HDL_PL: 91.72%
Total_CE: 91.72%
VLDL_CE: 91.72%
LDL_CE: 91.72%
HDL_CE: 91.72%
Total_FC: 91.72%
VLDL_FC: 91.72%
LDL_FC: 91.72%
HDL_FC: 91.72%
Total_L: 91.72%
VLDL_L: 91.72%
LDL_L: 91.72%
HDL_L: 91.72%
Total_P: 91.72%
VLDL_P: 91.72%
LDL_P: 91.72%
HDL_P: 91.72%
VLDL_size: 91.72%
LDL_size: 91.72%
HDL_size: 91.72%
Phosphoglyc: 91.72%
TG_by_PG: 91.72%
Cholines: 91.72%
Phosphatidylc: 91.72%
Sphingomyelins: 91.72%
ApoB: 91.72%
ApoA1: 91.72%
ApoB_by_ApoA1: 91.72%
Total_FA: 91.72%
Unsaturation: 91.72%
Omega_3: 91.72%
Omega_6: 91.72%
PUFA: 91.72%
MUFA: 91.72%
SFA: 91.72%
LA: 91.72%
DHA: 91.72%
Omega_3_pct: 91.72%
Omega_6_pct: 91.72%
PUFA_pct: 91.72%
MUFA_pct: 91.72%
SFA_pct: 91.72%
LA_pct: 91.72%
DHA_pct: 91.72%
PUFA_by_MUFA: 91.72%
Omega_6_by_Omega_3: 91.72%
Ala: 91.72%
Gln: 91.72%
Gly: 91.76%
His: 91.73%
Total_BCAA: 91.73%
Ile: 91.72%
Leu: 91.72%
Val: 91.73%
Phe: 91.72%
Tyr: 91.72%
Glucose: 91.74%
Lactate: 91.75%
Pyruvate: 91.73%
Citrate: 91.72%
bOHbutyrate: 91.90%
Acetate: 91.72%
Acetoacetate: 91.72%
Acetone: 91.72%
Creatinine: 91.74%
Albumin: 91.72%
GlycA: 91.72%
XXL_VLDL_P: 91.72%
XXL_VLDL_L: 91.72%
XXL_VLDL_PL: 91.72%
XXL_VLDL_C: 91.72%
XXL_VLDL_CE: 91.72%
XXL_VLDL_FC: 91.72%
XXL_VLDL_TG: 91.72%
XL_VLDL_P: 91.72%
XL_VLDL_L: 91.72%
XL_VLDL_PL: 91.72%
XL_VLDL_C: 91.72%
XL_VLDL_CE: 91.72%
XL_VLDL_FC: 91.72%
XL_VLDL_TG: 91.72%
L_VLDL_P: 91.72%
L_VLDL_L: 91.72%
L_VLDL_PL: 91.72%
L_VLDL_C: 91.72%
L_VLDL_CE: 91.72%
L_VLDL_FC: 91.72%
L_VLDL_TG: 91.72%
M_VLDL_P: 91.72%
M_VLDL_L: 91.72%
M_VLDL_PL: 91.72%
M_VLDL_C: 91.72%
M_VLDL_CE: 91.72%
M_VLDL_FC: 91.72%
M_VLDL_TG: 91.72%
S_VLDL_P: 91.72%
S_VLDL_L: 91.72%
S_VLDL_PL: 91.72%
S_VLDL_C: 91.72%
S_VLDL_CE: 91.72%
S_VLDL_FC: 91.72%
S_VLDL_TG: 91.72%
XS_VLDL_P: 91.72%
XS_VLDL_L: 91.72%
XS_VLDL_PL: 91.72%
XS_VLDL_C: 91.72%
XS_VLDL_CE: 91.72%
XS_VLDL_FC: 91.72%
XS_VLDL_TG: 91.72%
IDL_P: 91.72%
IDL_L: 91.72%
IDL_PL: 91.72%
IDL_C: 91.72%
IDL_CE: 91.72%
IDL_FC: 91.72%
IDL_TG: 91.72%
L_LDL_P: 91.72%
L_LDL_L: 91.72%
L_LDL_PL: 91.72%
L_LDL_C: 91.72%
L_LDL_CE: 91.72%
L_LDL_FC: 91.72%
L_LDL_TG: 91.72%
M_LDL_P: 91.72%
M_LDL_L: 91.72%
M_LDL_PL: 91.72%
M_LDL_C: 91.72%
M_LDL_CE: 91.72%
M_LDL_FC: 91.72%
M_LDL_TG: 91.72%
S_LDL_P: 91.72%
S_LDL_L: 91.72%
S_LDL_PL: 91.72%
S_LDL_C: 91.72%
S_LDL_CE: 91.72%
S_LDL_FC: 91.72%
S_LDL_TG: 91.72%
XL_HDL_P: 91.72%
XL_HDL_L: 91.72%
XL_HDL_PL: 91.72%
XL_HDL_C: 91.72%
XL_HDL_CE: 91.72%
XL_HDL_FC: 91.72%
XL_HDL_TG: 91.72%
L_HDL_P: 91.72%
L_HDL_L: 91.72%
L_HDL_PL: 91.72%
L_HDL_C: 91.72%
L_HDL_CE: 91.72%
L_HDL_FC: 91.72%
L_HDL_TG: 91.72%
M_HDL_P: 91.72%
M_HDL_L: 91.72%
M_HDL_PL: 91.72%
M_HDL_C: 91.72%
M_HDL_CE: 91.72%
M_HDL_FC: 91.72%
M_HDL_TG: 91.72%
S_HDL_P: 91.72%
S_HDL_L: 91.72%
S_HDL_PL: 91.72%
S_HDL_C: 91.72%
S_HDL_CE: 91.72%
S_HDL_FC: 91.72%
S_HDL_TG: 91.72%
XXL_VLDL_PL_pct: 92.13%
XXL_VLDL_C_pct: 92.13%
XXL_VLDL_CE_pct: 92.13%
XXL_VLDL_FC_pct: 92.13%
XXL_VLDL_TG_pct: 92.13%
XL_VLDL_PL_pct: 91.73%
XL_VLDL_C_pct: 91.73%
XL_VLDL_CE_pct: 91.73%
XL_VLDL_FC_pct: 91.73%
XL_VLDL_TG_pct: 91.73%
L_VLDL_PL_pct: 91.72%
L_VLDL_C_pct: 91.72%
L_VLDL_CE_pct: 91.72%
L_VLDL_FC_pct: 91.72%
L_VLDL_TG_pct: 91.72%
M_VLDL_PL_pct: 91.72%
M_VLDL_C_pct: 91.72%
M_VLDL_CE_pct: 91.72%
M_VLDL_FC_pct: 91.72%
M_VLDL_TG_pct: 91.72%
S_VLDL_PL_pct: 91.72%
S_VLDL_C_pct: 91.72%
S_VLDL_CE_pct: 91.72%
S_VLDL_FC_pct: 91.72%
S_VLDL_TG_pct: 91.72%
XS_VLDL_PL_pct: 91.72%
XS_VLDL_C_pct: 91.72%
XS_VLDL_CE_pct: 91.72%
XS_VLDL_FC_pct: 91.72%
XS_VLDL_TG_pct: 91.72%
IDL_PL_pct: 91.72%
IDL_C_pct: 91.72%
IDL_CE_pct: 91.72%
IDL_FC_pct: 91.72%
IDL_TG_pct: 91.72%
L_LDL_PL_pct: 91.72%
L_LDL_C_pct: 91.72%
L_LDL_CE_pct: 91.72%
L_LDL_FC_pct: 91.72%
L_LDL_TG_pct: 91.72%
M_LDL_PL_pct: 91.72%
M_LDL_C_pct: 91.72%
M_LDL_CE_pct: 91.72%
M_LDL_FC_pct: 91.72%
M_LDL_TG_pct: 91.72%
S_LDL_PL_pct: 91.72%
S_LDL_C_pct: 91.72%
S_LDL_CE_pct: 91.72%
S_LDL_FC_pct: 91.72%
S_LDL_TG_pct: 91.72%
XL_HDL_PL_pct: 91.72%
XL_HDL_C_pct: 91.72%
XL_HDL_CE_pct: 91.72%
XL_HDL_FC_pct: 91.72%
XL_HDL_TG_pct: 91.72%
L_HDL_PL_pct: 91.72%
L_HDL_C_pct: 91.72%
L_HDL_CE_pct: 91.72%
L_HDL_FC_pct: 91.72%
L_HDL_TG_pct: 91.72%
M_HDL_PL_pct: 91.72%
M_HDL_C_pct: 91.72%
M_HDL_CE_pct: 91.72%
M_HDL_FC_pct: 91.72%
M_HDL_TG_pct: 91.72%
S_HDL_PL_pct: 91.72%
S_HDL_C_pct: 91.72%
S_HDL_CE_pct: 91.72%
S_HDL_FC_pct: 91.72%
S_HDL_TG_pct: 91.72%
cancer: 79.32%

"""

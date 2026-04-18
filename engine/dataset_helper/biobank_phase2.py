from engine.dataset_helper.base import *


class BiobankSensitiveDataset(EvaluatedDataset):
    def __init__(
        self,
        cancer,
        target="has_cancer",
        notebook_path=None,
        is_encode=True,
        is_metabolomics=False,
    ):
        super().__init__(notebook_path)

        self.target = target
        self.cancer = cancer
        self.output = "classification"
        self.ref = "1800-01-01"
        self.is_metabolomics = is_metabolomics

        src = "database/dataset/biobank_sensitive"
        folder = self._get_dataset_folder()[0]
        path_train = f"database/dataset/{folder}/df_merge.csv"

        dst = src.replace("biobank_sensitive", folder)

        print(f">> copying {src} to {dst}")
        self._copy_folder(src, dst)

        self.path_train = self._get_path(path_train)
        self.data = self._read_train()
        self._preprocess()

        ## Afer discussion with Bea and Calle 20241118
        if self.is_metabolomics:
            self.data = self.data.drop(
                columns=[
                    ## drop many missing values
                    "exclude_diet_2",
                    "inclusion_age_2",
                    "fasting_q_2",
                    "height_2",
                    "weight_2",
                    "bmi_2",
                    "waist_2",
                    "total_cholesterol_2",
                    "hdl_2",
                    "ldl_2",
                    "triglycerides_2",
                    "bloods_0h_2",
                    "bloods_2h_2",
                    "systolic_bp_2",
                    "diastolic_bp_2",
                    "marital_status_2",
                    "education_2",
                    "eventdate_2",
                    "exclude_diet_3",
                    "inclusion_age_3",
                    "fasting_q_3",
                    "height_3",
                    "weight_3",
                    "bmi_3",
                    "waist_3",
                    "total_cholesterol_3",
                    "hdl_3",
                    "ldl_3",
                    "triglycerides_3",
                    "bloods_0h_3",
                    "bloods_2h_3",
                    "systolic_bp_3",
                    "diastolic_bp_3",
                    "marital_status_3",
                    "education_3",
                    "eventdate_3",
                ]
            )
            self.data = self.data[
                self.data["S_HDL_FC_pct"] != -1
            ]  # keep only records with metabolomics
            self.data = self.data.reset_index(drop=True)
        else:
            self.data = self.data.drop(
                columns=[
                    ## drop many missing values
                    "exclude_diet_2",
                    "inclusion_age_2",
                    "fasting_q_2",
                    "height_2",
                    "weight_2",
                    "bmi_2",
                    "waist_2",
                    "total_cholesterol_2",
                    "hdl_2",
                    "ldl_2",
                    "triglycerides_2",
                    "bloods_0h_2",
                    "bloods_2h_2",
                    "systolic_bp_2",
                    "diastolic_bp_2",
                    "marital_status_2",
                    "education_2",
                    "eventdate_2",
                    "exclude_diet_3",
                    "inclusion_age_3",
                    "fasting_q_3",
                    "height_3",
                    "weight_3",
                    "bmi_3",
                    "waist_3",
                    "total_cholesterol_3",
                    "hdl_3",
                    "ldl_3",
                    "triglycerides_3",
                    "bloods_0h_3",
                    "bloods_2h_3",
                    "systolic_bp_3",
                    "diastolic_bp_3",
                    "marital_status_3",
                    "education_3",
                    "eventdate_3",
                    "EDTA_plasma",
                    "Citrate_plasma",
                    "Low_ethanol",
                    "Medium_ethanol",
                    "High_ethanol",
                    "Isopropyl_alcohol",
                    "N_methyl_2_pyrrolidone",
                    "Polysaccharides",
                    "Aminocaproic_acid",
                    "Low_glucose",
                    "High_lactate",
                    "High_pyruvate",
                    "Low_glutamine_or_high_glutamate",
                    "Gluconolactone",
                    "Low_protein",
                    "Unexpected_amino_acid_signals",
                    "Unidentified_macromolecules",
                    "Unidentified_small_molecule_a",
                    "Unidentified_small_molecule_b",
                    "Unidentified_small_molecule_c",
                    "Below_limit_of_quantification",
                    "Total_C",
                    "non_HDL_C",
                    "Remnant_C",
                    "VLDL_C",
                    "Clinical_LDL_C",
                    "LDL_C",
                    "HDL_C",
                    "Total_TG",
                    "VLDL_TG",
                    "LDL_TG",
                    "HDL_TG",
                    "Total_PL",
                    "VLDL_PL",
                    "LDL_PL",
                    "HDL_PL",
                    "Total_CE",
                    "VLDL_CE",
                    "LDL_CE",
                    "HDL_CE",
                    "Total_FC",
                    "VLDL_FC",
                    "LDL_FC",
                    "HDL_FC",
                    "Total_L",
                    "VLDL_L",
                    "LDL_L",
                    "HDL_L",
                    "Total_P",
                    "VLDL_P",
                    "LDL_P",
                    "HDL_P",
                    "VLDL_size",
                    "LDL_size",
                    "HDL_size",
                    "Phosphoglyc",
                    "TG_by_PG",
                    "Cholines",
                    "Phosphatidylc",
                    "Sphingomyelins",
                    "ApoB",
                    "ApoA1",
                    "ApoB_by_ApoA1",
                    "Total_FA",
                    "Unsaturation",
                    "Omega_3",
                    "Omega_6",
                    "PUFA",
                    "MUFA",
                    "SFA",
                    "LA",
                    "DHA",
                    "Omega_3_pct",
                    "Omega_6_pct",
                    "PUFA_pct",
                    "MUFA_pct",
                    "SFA_pct",
                    "LA_pct",
                    "DHA_pct",
                    "PUFA_by_MUFA",
                    "Omega_6_by_Omega_3",
                    "Ala",
                    "Gln",
                    "Gly",
                    "His",
                    "Total_BCAA",
                    "Ile",
                    "Leu",
                    "Val",
                    "Phe",
                    "Tyr",
                    "Glucose",
                    "Lactate",
                    "Pyruvate",
                    "Citrate",
                    "bOHbutyrate",
                    "Acetate",
                    "Acetoacetate",
                    "Acetone",
                    "Creatinine",
                    "Albumin",
                    "GlycA",
                    "XXL_VLDL_P",
                    "XXL_VLDL_L",
                    "XXL_VLDL_PL",
                    "XXL_VLDL_C",
                    "XXL_VLDL_CE",
                    "XXL_VLDL_FC",
                    "XXL_VLDL_TG",
                    "XL_VLDL_P",
                    "XL_VLDL_L",
                    "XL_VLDL_PL",
                    "XL_VLDL_C",
                    "XL_VLDL_CE",
                    "XL_VLDL_FC",
                    "XL_VLDL_TG",
                    "L_VLDL_P",
                    "L_VLDL_L",
                    "L_VLDL_PL",
                    "L_VLDL_C",
                    "L_VLDL_CE",
                    "L_VLDL_FC",
                    "L_VLDL_TG",
                    "M_VLDL_P",
                    "M_VLDL_L",
                    "M_VLDL_PL",
                    "M_VLDL_C",
                    "M_VLDL_CE",
                    "M_VLDL_FC",
                    "M_VLDL_TG",
                    "S_VLDL_P",
                    "S_VLDL_L",
                    "S_VLDL_PL",
                    "S_VLDL_C",
                    "S_VLDL_CE",
                    "S_VLDL_FC",
                    "S_VLDL_TG",
                    "XS_VLDL_P",
                    "XS_VLDL_L",
                    "XS_VLDL_PL",
                    "XS_VLDL_C",
                    "XS_VLDL_CE",
                    "XS_VLDL_FC",
                    "XS_VLDL_TG",
                    "IDL_P",
                    "IDL_L",
                    "IDL_PL",
                    "IDL_C",
                    "IDL_CE",
                    "IDL_FC",
                    "IDL_TG",
                    "L_LDL_P",
                    "L_LDL_L",
                    "L_LDL_PL",
                    "L_LDL_C",
                    "L_LDL_CE",
                    "L_LDL_FC",
                    "L_LDL_TG",
                    "M_LDL_P",
                    "M_LDL_L",
                    "M_LDL_PL",
                    "M_LDL_C",
                    "M_LDL_CE",
                    "M_LDL_FC",
                    "M_LDL_TG",
                    "S_LDL_P",
                    "S_LDL_L",
                    "S_LDL_PL",
                    "S_LDL_C",
                    "S_LDL_CE",
                    "S_LDL_FC",
                    "S_LDL_TG",
                    "XL_HDL_P",
                    "XL_HDL_L",
                    "XL_HDL_PL",
                    "XL_HDL_C",
                    "XL_HDL_CE",
                    "XL_HDL_FC",
                    "XL_HDL_TG",
                    "L_HDL_P",
                    "L_HDL_L",
                    "L_HDL_PL",
                    "L_HDL_C",
                    "L_HDL_CE",
                    "L_HDL_FC",
                    "L_HDL_TG",
                    "M_HDL_P",
                    "M_HDL_L",
                    "M_HDL_PL",
                    "M_HDL_C",
                    "M_HDL_CE",
                    "M_HDL_FC",
                    "M_HDL_TG",
                    "S_HDL_P",
                    "S_HDL_L",
                    "S_HDL_PL",
                    "S_HDL_C",
                    "S_HDL_CE",
                    "S_HDL_FC",
                    "S_HDL_TG",
                    "XXL_VLDL_PL_pct",
                    "XXL_VLDL_C_pct",
                    "XXL_VLDL_CE_pct",
                    "XXL_VLDL_FC_pct",
                    "XXL_VLDL_TG_pct",
                    "XL_VLDL_PL_pct",
                    "XL_VLDL_C_pct",
                    "XL_VLDL_CE_pct",
                    "XL_VLDL_FC_pct",
                    "XL_VLDL_TG_pct",
                    "L_VLDL_PL_pct",
                    "L_VLDL_C_pct",
                    "L_VLDL_CE_pct",
                    "L_VLDL_FC_pct",
                    "L_VLDL_TG_pct",
                    "M_VLDL_PL_pct",
                    "M_VLDL_C_pct",
                    "M_VLDL_CE_pct",
                    "M_VLDL_FC_pct",
                    "M_VLDL_TG_pct",
                    "S_VLDL_PL_pct",
                    "S_VLDL_C_pct",
                    "S_VLDL_CE_pct",
                    "S_VLDL_FC_pct",
                    "S_VLDL_TG_pct",
                    "XS_VLDL_PL_pct",
                    "XS_VLDL_C_pct",
                    "XS_VLDL_CE_pct",
                    "XS_VLDL_FC_pct",
                    "XS_VLDL_TG_pct",
                    "IDL_PL_pct",
                    "IDL_C_pct",
                    "IDL_CE_pct",
                    "IDL_FC_pct",
                    "IDL_TG_pct",
                    "L_LDL_PL_pct",
                    "L_LDL_C_pct",
                    "L_LDL_CE_pct",
                    "L_LDL_FC_pct",
                    "L_LDL_TG_pct",
                    "M_LDL_PL_pct",
                    "M_LDL_C_pct",
                    "M_LDL_CE_pct",
                    "M_LDL_FC_pct",
                    "M_LDL_TG_pct",
                    "S_LDL_PL_pct",
                    "S_LDL_C_pct",
                    "S_LDL_CE_pct",
                    "S_LDL_FC_pct",
                    "S_LDL_TG_pct",
                    "XL_HDL_PL_pct",
                    "XL_HDL_C_pct",
                    "XL_HDL_CE_pct",
                    "XL_HDL_FC_pct",
                    "XL_HDL_TG_pct",
                    "L_HDL_PL_pct",
                    "L_HDL_C_pct",
                    "L_HDL_CE_pct",
                    "L_HDL_FC_pct",
                    "L_HDL_TG_pct",
                    "M_HDL_PL_pct",
                    "M_HDL_C_pct",
                    "M_HDL_CE_pct",
                    "M_HDL_FC_pct",
                    "M_HDL_TG_pct",
                    "S_HDL_PL_pct",
                    "S_HDL_C_pct",
                    "S_HDL_CE_pct",
                    "S_HDL_FC_pct",
                    "S_HDL_TG_pct",
                ]
            )

        self.type_columns = self._get_type_columns()
        self.columns = list(self.type_columns.keys())
        self.features = []
        self.discrete_columns = []
        self._setup_task()

        for col in self.data.columns:
            if col in self.discrete_columns:
                # print(f">> converting {col} to str")
                self.data[col] = self.data[col].astype(str)
            else:
                # print(f">> converting {col} to float")
                self.data[col] = pd.to_numeric(self.data[col], errors="coerce").astype(
                    float
                )

        print(self.data)

        self.data.to_csv(
            f"database/dataset/{folder}/data_all.csv",
            sep="\t",
            encoding="utf-8",
        )

        self._split_df_save_index()

        if is_encode:
            self._encode_label()

        print(self.data_train)

        self._drop_duplicates()

        self._prep_ctab()
        self._prep_tabddpm()
        self._prep_tabddpm_config_toml_mlp()
        self._prep_tabddpm_config_toml_resnet()
        self._prep_tabsyn()

        self._init_key_sensitive_fields()

    def _print_type(self, df, col):
        types_in_column_A = df[col].apply(type).unique()
        print(types_in_column_A)

    def _read_train(self) -> pd.DataFrame:
        return pd.read_csv(
            self.path_train,
            sep="\t",
            header=0,
            index_col=0,
        )

    def _read_test(self):
        pass

    def _get_type_columns(self) -> dict:
        list_discrete = [
            "dead_by_april2022",
            "sex",
            "exclude_diet",
            "fasting_q",
            "marital_status",
            "education",
            "diagnosis_place_of_residence",
            "lkf_value",
            "malignancy_status",
            "figo_value",
            "t_value",
            "m_value",
            "n_value",
            "EDTA_plasma",
            "Citrate_plasma",
            "Low_ethanol",
            "Medium_ethanol",
            "High_ethanol",
            "Isopropyl_alcohol",
            "N_methyl_2_pyrrolidone",
            "Polysaccharides",
            "Aminocaproic_acid",
            "Low_glucose",
            "High_lactate",
            "High_pyruvate",
            "Low_glutamine_or_high_glutamate",
            "Gluconolactone",
            "Low_protein",
            "Unexpected_amino_acid_signals",
            "Unidentified_macromolecules",
            "Unidentified_small_molecule_a",
            "Unidentified_small_molecule_b",
            "Unidentified_small_molecule_c",
            "Below_limit_of_quantification",
        ]

        if self.is_metabolomics:
            list_discrete.append("has_cancer")

        t = {}
        for c in list(self.data.columns):
            if any(d in c for d in list_discrete):
                t[c] = "discrete"
            else:
                t[c] = "continuous"
        # print(t)
        return t

    def _preprocess(self):
        # Replace "TAG" variations with -1
        self.data = self.data.replace(r"(?i)\btag\b", -1, regex=True)

        if self.is_metabolomics:
            self.data = self.generate_meta_df(self.data)
        else:
            self.data = self.generate_cancer_df(self.data)

    def generate_cancer_df(self, df_merge):
        # Define cancer types and corresponding filenames
        cancer_types = {
            "prostate": "prostate",
            "breast": "breast",
            "colorectal": "colorectal",
            "urothelial and kidney": "uroandkid",
            "lung": "lung",
            "pancreatic": "pancreatic",
            "haematological": "haematological",
        }

        replace_values = [-1, "-1.0", "-1"]

        # Reference date for conversion
        reference_date = pd.Timestamp("1800-01-01")

        # Convert date columns to the number of days since reference date
        for column in df_merge.columns:
            if "date" in column.lower():
                # Replace specified values with -1
                df_merge[column] = df_merge[column].apply(
                    lambda x: -1 if pd.isna(x) or x in replace_values else x
                )

                # Convert the valid date strings to datetime and calculate days since reference date
                df_merge[column] = pd.to_datetime(df_merge[column], errors="coerce")
                df_merge[column] = (df_merge[column] - reference_date) // pd.Timedelta(
                    "1D"
                )

        # Iterate through each cancer type
        for k, v in cancer_types.items():
            if v == self.cancer:
                # Create a boolean series where True if the patient has the given cancer type
                cancer_df = df_merge.copy()
                cancer_df["has_cancer"] = (cancer_df["cancer"] == k).astype(int)
                cancer_df = cancer_df.drop(["cancer"], axis=1)
                return cancer_df

    def generate_meta_df(self, df_merge):
        replace_values = [-1, "-1.0", "-1"]

        # Reference date for conversion
        reference_date = pd.Timestamp("1800-01-01")

        # Convert date columns to the number of days since reference date
        for column in df_merge.columns:
            if "date" in column.lower():
                # Replace specified values with -1
                df_merge[column] = df_merge[column].apply(
                    lambda x: -1 if pd.isna(x) or x in replace_values else x
                )

                # Convert the valid date strings to datetime and calculate days since reference date
                df_merge[column] = pd.to_datetime(df_merge[column], errors="coerce")
                df_merge[column] = (df_merge[column] - reference_date) // pd.Timedelta(
                    "1D"
                )

        cancer_df = df_merge.copy()
        cancer_df["has_cancer"] = cancer_df["cancer"]
        cancer_df = cancer_df.drop(["cancer"], axis=1)

        return cancer_df

    def _prep_ctab(self):
        self.categorical_columns = self.discrete_columns
        self.log_columns = []
        self.mixed_columns = {}
        self.general_columns = []
        self.non_categorical_columns = []
        self.integer_columns = []
        self.problem_type = {"Classification": self.target}

    def _init_key_sensitive_fields(self):
        self.key_fields = [
            "sex",
            "age",
            "diagnosis_date",
            "diagnosis_place_of_residence",
            "lkf_value",
            "marital_status_5",
            "education_5",
        ]

        self.sensitive_fields = [
            "has_cancer",
            "dead_by_april2022",
            "malignancy_status",
            "figo_value",
            "t_value",
            "m_value",
            "n_value",
            "eventdate",
            "bmi_5",
            "waist_5",
            "total_cholesterol_5",
            "hdl_5",
            "ldl_5",
            "triglycerides_5",
            "bloods_0h_5",
            "bloods_2h_5",
            "systolic_bp_5",
            "diastolic_bp_5",
            "height_5",
            "weight_5",
        ]


class BiobankSensitiveProstateDataset(BiobankSensitiveDataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer="prostate", notebook_path=notebook_path, is_encode=is_encode
        )


class BiobankSensitiveBreastDataset(BiobankSensitiveDataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer="breast", notebook_path=notebook_path, is_encode=is_encode
        )


class BiobankSensitiveColorectalDataset(BiobankSensitiveDataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer="colorectal", notebook_path=notebook_path, is_encode=is_encode
        )


class BiobankSensitiveUrothelialAndKidneyDataset(BiobankSensitiveDataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer="uroandkid", notebook_path=notebook_path, is_encode=is_encode
        )


class BiobankSensitiveLungDataset(BiobankSensitiveDataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer="lung", notebook_path=notebook_path, is_encode=is_encode
        )


class BiobankSensitivePancreaticDataset(BiobankSensitiveDataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer="pancreatic", notebook_path=notebook_path, is_encode=is_encode
        )


class BiobankSensitiveHaematologicalDataset(BiobankSensitiveDataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer="haematological", notebook_path=notebook_path, is_encode=is_encode
        )


class BiobankSensitiveMetabolomicsDataset(BiobankSensitiveDataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer=None,
            notebook_path=notebook_path,
            is_encode=is_encode,
            is_metabolomics=True,
        )

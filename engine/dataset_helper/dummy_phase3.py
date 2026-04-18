from engine.dataset_helper.base import *


class DummyPhase3Dataset(EvaluatedDataset):
    def __init__(
        self,
        cancer,
        target="cancer",
        notebook_path=None,
        is_encode=True,
        is_encode_missing=False,
        is_small=False,
        is_pca=False,
    ):
        super().__init__(notebook_path)

        dict_file = {
            "biobank_phase3_dummy": "dummy.csv",
            "biobank_phase3_dummy_missing": "dummy.csv",
            "biobank_phase3_dummy_small": "dummy.csv",
            "biobank_phase3_dummy_missing_small": "dummy.csv",
            "biobank_phase3_dummy_pca": "dummy.csv",
        }

        self.target = target
        self.cancer = cancer
        self.output = "classification"
        self.ref = "1800-01-01"
        self.is_encode_missing = is_encode_missing
        self.is_pca = is_pca

        # Override or modify the setup
        folder = self._get_dataset_folder()[0]
        filename = dict_file[cancer]

        if os.path.exists(
            "/cygnus/proj_nobackup/wharf/minhvu/minhvu-sens2024537/biobank-anonymization/"
        ):
            self.notebook_path = "/cygnus/proj_nobackup/wharf/minhvu/minhvu-sens2024537/biobank-anonymization/"

        src_file = os.path.join("database/dataset/daniel/", filename)
        dst_dir = os.path.join("database/dataset/", folder)

        src_file = self._get_path(src_file)
        dst_dir = self._get_path(dst_dir)

        path_train = os.path.join("database/dataset/", folder, filename)
        self.path_train = self._get_path(path_train)

        print(src_file)
        print(path_train)

        print(f">> copying {src_file} to {path_train}")
        self._copy_file(src_file, dst_dir)

        self.data = self._read_train()
        if is_small:
            self.data = self.data[
                [
                    "gender",
                    "age",
                    "birthdate",
                    "accessdate",
                    "height",
                    "weight",
                    "bmi",
                    "ap_hi",
                    "ap_lo",
                    "alco",
                    "active",
                    "cholesterol",
                    "gluc",
                    "smoke",
                    "cardio",
                ]
            ].copy()

        self.type_columns = self._get_type_columns()

        if is_encode_missing:
            self._preprocess_missing()
        else:
            self._preprocess()

        if is_pca:
            self.perform_pca()

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

        data_all_path = f"database/dataset/{folder}/data_all.csv"
        data_all_path = self._get_path(data_all_path)
        self.data.to_csv(
            data_all_path,
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
        return pd.read_csv(self.path_train, sep="\t", header=0)

    def _read_test(self):
        pass

    def _get_type_columns(self, verbose=False) -> dict:
        """
        Ignores missing values (np.nan) when counting unique values
        If number of unique non-null values < 15:
        ➔ set dtype to a number (like float)

        Else:
        ➔ set dtype to object (so later you can treat it as categorical/discrete).
        """
        # lower case
        self.data = self.data.applymap(
            lambda x: x.strip().lower() if isinstance(x, str) else x
        )

        cont_cols_input, date_cols_input, str_cols_input, mixed_cols_input = (
            self._categorize_columns_from_input(self.data)
        )

        for col in self.data.columns:
            n_unique = self.data[col].dropna().nunique()

            if n_unique > 15 and col in cont_cols_input:
                # Set to numeric type (float)
                if verbose:
                    print(f"{col} is cont.")
                self.data[col] = pd.to_numeric(self.data[col], errors="coerce")
            else:
                # Set to object type (treat as discrete/categorical later)
                if verbose:
                    print(f"{col} is cat.")
                self.data[col] = self.data[col].astype("object")

        list_discrete = self.data.select_dtypes(include=["object"]).columns.tolist()
        list_discrete.append(self.target)

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
        self.data = self.data.applymap(
            lambda x: x.strip().lower() if isinstance(x, str) else x
        )
        replace_values = [-1, "-1.0", "-1"]

        # Reference date for conversion
        reference_date = pd.Timestamp("1800-01-01")

        # Convert date columns to the number of days since reference date
        for column in self.data.columns:
            if "date" in column.lower():
                # Replace specified values with -1
                self.data[column] = self.data[column].apply(
                    lambda x: -1 if pd.isna(x) or x in replace_values else x
                )

                # Convert the valid date strings to datetime and calculate days since reference date
                self.data[column] = pd.to_datetime(self.data[column], errors="coerce")
                self.data[column] = (
                    self.data[column] - reference_date
                ) // pd.Timedelta("1D")
                self.data[column] = self.data[column].astype(float)
                self.type_columns[column] = "continuous"
            elif self.data[column].dtype == "object":
                self.data[column] = self.data[column].fillna("-1")
            else:
                self.data[column] = self.data[column].fillna(-1)
        self.data[self.target] = self.data[self.target].astype(
            int
        )  # convert True False to 1 and 0

    def _preprocess_missing(self):
        """
        Preprocess missing values in the dataset.

        Steps:
        1. Convert all date columns to the number of days since a fixed reference date.
        2. For discrete variables:
            - Fill missing values with the string "-1".
        3. For continuous variables:
            - Add a binary missing indicator column.
            - Impute missing values using the median, optionally adding Gaussian noise.
            - Standardize the column (zero mean, unit variance).
        4. Convert target variable to integer.

        The function also stores mean and std of original continuous variables for potential inverse transform.
        """
        self.data_original_continuous_info = {}
        reference_date = pd.Timestamp("1800-01-01")

        # Step 1: Convert date columns to days since reference date
        for column in self.data.columns:
            if "date" in column.lower():
                self.data[column] = pd.to_datetime(self.data[column], errors="coerce")
                self.data[column] = (
                    self.data[column] - reference_date
                ) // pd.Timedelta("1D")
                self.data[column] = self.data[column].astype(float)
                self.type_columns[column] = "continuous"

        # Step 2–5: Handle missing values based on column type
        for column in self.data.columns:
            col_type = self.type_columns.get(column)

            if col_type == "discrete":
                # Fill missing discrete values with sentinel string "-1"
                self.data[column] = self.data[column].fillna("-1")

            elif col_type == "continuous":
                has_nan = self.data[column].isna().any()
                if has_nan:
                    # Step 2a: Replace NaN with -1 temporarily
                    self.data[column] = self.data[column].fillna(-1)

                    # Step 2b: Add binary indicator for missingness
                    missing_flag_col = f"{column}_missing"
                    self.data[missing_flag_col] = (self.data[column] == -1).astype(int)
                    self.type_columns[missing_flag_col] = "discrete"

                    # Step 2c: Impute -1 values using median + optional noise
                    median_val = self.data.loc[self.data[column] != -1, column].median()
                    missing_idx = self.data[column] == -1
                    noise_std = 0.05

                    if noise_std > 0 and missing_idx.any():
                        noise = np.random.normal(
                            loc=0, scale=noise_std, size=missing_idx.sum()
                        )
                        self.data.loc[missing_idx, column] = median_val + noise
                    else:
                        self.data.loc[missing_idx, column] = median_val

                # Step 3: We do not Standardize -- instead we use quantiletransformer for better mapping

        # Step 6: Ensure target variable is of integer type
        self.data[self.target] = self.data[self.target].astype(int)

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
            "gender",
            "age",
            "birthdate",
            "accessdate",
        ]

        self.sensitive_fields = [
            "height",
            "weight",
            "bmi",
            "ap_hi",
            "ap_lo",
            "cholesterol",
            "gluc",
            "smoke",
        ]


class BiobankPhase3DummyDataset(DummyPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True, target="cardio"):
        super().__init__(
            cancer="biobank_phase3_dummy",
            notebook_path=notebook_path,
            is_encode=is_encode,
            target=target,
            is_encode_missing=False,
            is_small=False,
        )


class BiobankPhase3DummyMissingDataset(DummyPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True, target="cardio"):
        super().__init__(
            cancer="biobank_phase3_dummy_missing",
            notebook_path=notebook_path,
            is_encode=is_encode,
            target=target,
            is_encode_missing=True,
            is_small=False,
        )


class BiobankPhase3DummySmallDataset(DummyPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True, target="cardio"):
        super().__init__(
            cancer="biobank_phase3_dummy_small",
            notebook_path=notebook_path,
            is_encode=is_encode,
            target=target,
            is_encode_missing=False,
            is_small=True,
        )


class BiobankPhase3DummyMissingSmallDataset(DummyPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True, target="cardio"):
        super().__init__(
            cancer="biobank_phase3_dummy_missing_small",
            notebook_path=notebook_path,
            is_encode=is_encode,
            target=target,
            is_encode_missing=True,
            is_small=True,
        )


class BiobankPhase3DummySmallDataset(DummyPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True, target="cardio"):
        super().__init__(
            cancer="biobank_phase3_dummy_small",
            notebook_path=notebook_path,
            is_encode=is_encode,
            target=target,
            is_encode_missing=False,
            is_small=True,
        )


class BiobankPhase3DummyPCADataset(DummyPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True, target="cardio"):
        super().__init__(
            cancer="biobank_phase3_dummy_pca",
            notebook_path=notebook_path,
            is_encode=is_encode,
            target=target,
            is_encode_missing=True,
            is_small=False,
            is_pca=True,
        )


def main():
    s = {
        # "biobank_phase3_dummy": BiobankPhase3DummyDataset,
        # "biobank_phase3_dummy_missing": BiobankPhase3DummyMissingDataset,
        # "biobank_phase3_dummy_small": BiobankPhase3DummySmallDataset,
        # "biobank_phase3_dummy_missing_small": BiobankPhase3DummyMissingSmallDataset,
        "biobank_phase3_dummy_pca": BiobankPhase3DummyPCADataset,
    }

    for k, v in s.items():
        print(k, v)
        d = globals()[DATASET_CLASSES[k]]()
        print(d.data)

        print(d.data_train)
        print(d.data_train.describe())
        print(d.data_test)
        print("features:", d.features)
        print("discrete:", d.discrete_columns)
        print("target:", d.target)
        print(len(d.discrete_columns))
        print(len(d.continuous_columns))
        print(len(d.features))

        print(d.key_fields)
        print(d.sensitive_fields)


if __name__ == "__main__":
    main()

from engine.dataset_helper.base import *
from engine.dataset_helper.preprocessing import *


class BiobankPhase3Dataset(EvaluatedDataset):
    def __init__(
        self,
        cancer,
        target="cancer",
        notebook_path=None,
        is_encode=True,
        is_encode_missing=True,
        is_pca=False,
    ):
        super().__init__(notebook_path)

        dict_file = {
            "biobank_phase3_cancer_150": "nshds_cancer_150.tsv",
            "biobank_phase3_cancer_151": "nshds_cancer_151.tsv",
            "biobank_phase3_cancer_153": "nshds_cancer_153.tsv",
            "biobank_phase3_cancer_154": "nshds_cancer_154.tsv",
            "biobank_phase3_cancer_155": "nshds_cancer_155.tsv",
            "biobank_phase3_cancer_156": "nshds_cancer_156.tsv",
            "biobank_phase3_cancer_157": "nshds_cancer_157.tsv",
            "biobank_phase3_cancer_162": "nshds_cancer_162.tsv",
            "biobank_phase3_cancer_172": "nshds_cancer_172.tsv",
            "biobank_phase3_cancer_173": "nshds_cancer_173.tsv",
            "biobank_phase3_cancer_174": "nshds_cancer_174.tsv",
            "biobank_phase3_cancer_180": "nshds_cancer_180.tsv",
            "biobank_phase3_cancer_182": "nshds_cancer_182.tsv",
            "biobank_phase3_cancer_183": "nshds_cancer_183.tsv",
            "biobank_phase3_cancer_185": "nshds_cancer_185.tsv",
            "biobank_phase3_cancer_188": "nshds_cancer_188.tsv",
            "biobank_phase3_cancer_189": "nshds_cancer_189.tsv",
            "biobank_phase3_cancer_191": "nshds_cancer_191.tsv",
            "biobank_phase3_cancer_192": "nshds_cancer_192.tsv",
            "biobank_phase3_cancer_193": "nshds_cancer_193.tsv",
            "biobank_phase3_cancer_194": "nshds_cancer_194.tsv",
            "biobank_phase3_cancer_199": "nshds_cancer_199.tsv",
            "biobank_phase3_cancer_200": "nshds_cancer_200.tsv",
            "biobank_phase3_cancer_203": "nshds_cancer_203.tsv",
            "biobank_phase3_cancer_204": "nshds_cancer_204.tsv",
            "biobank_phase3_cancer_205": "nshds_cancer_205.tsv",
            "biobank_phase3_cancer_all": "nshds_cancer_all.tsv",
            "biobank_phase3_cancer_bio_all": "nshds_cancer_bio_all.tsv",
            # dummy
            "biobank_phase3_dummy": "dummy.csv",
            "biobank_phase3_dummy_pca": "dummy.csv",
            "biobank_phase3_dummy_multi_pca": "dummy_multi.csv",
        }

        self.target = target
        self.cancer = cancer
        self.output = "classification"
        self.ref = "1800-01-01"

        # get pca excluded columns
        self.pca_excluded_columns = [
            self.target,
            "sex",
            "gender",
            "marital_status",
            "age",
            "height",
            "weight",
            "waist",
            "bmi",
        ]

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

        self.data, replaced_count = self._replace_lowercase_tag_with_nan_and_count(
            self.data
        )
        self.data_original = self.data.copy()

        print("total count of 'tag':", replaced_count)

        self.type_columns = self._get_type_columns()

        self.pipeline = FlexiblePipeline(
            [
                ("date", DateEncoder()),
                ("missing", MissingValueEncoder()),
                ("binary", BinaryColumnEncoder()),
                ("preprocess", DataFramePreprocessor(use_ohe_for_discrete=False)),
            ],
            type_columns=self.type_columns,
        )
        self.data = self.pipeline.fit_transform(self.data)
        self.type_columns = self.pipeline.type_columns

        """
        if is_pca:
            if "dummy" in self.cancer:
                self.perform_pca(var_threshold=0.8)
            else:
                self.perform_pca(var_threshold=0.9)
        """

        # if self.output == "regression":
        #     self.data[self.target] = self.data[self.target].astype(float)
        #     self.type_columns[self.target] = "continuous"
        # else:
        #     self.data[self.target] = self.data[self.target].astype(float)
        #     self.type_columns[self.target] = "continuous"

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

        # print(self.data)

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

            data_all_path = f"database/dataset/{folder}/data_all_encoded.csv"
            data_all_path = self._get_path(data_all_path)
            self.data.to_csv(
                data_all_path,
                sep="\t",
                encoding="utf-8",
            )

        print(self.data_train)

        data_all_path = f"database/dataset/{folder}/preprocessed.csv"
        data_all_path = self._get_path(data_all_path)
        self.data_train.to_csv(
            data_all_path,
            sep="\t",
            encoding="utf-8",
        )

        self._drop_duplicates()

        self._prep_ctab()
        self._prep_tabddpm()
        self._prep_tabddpm_config_toml_mlp()
        self._prep_tabddpm_config_toml_resnet()
        self._prep_tabsyn()

        if "dummy" in self.cancer:
            self._init_key_sensitive_fields_dummy()
        else:
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

            is_continuous = (
                n_unique > 15
                and col in cont_cols_input
                and not (
                    col in cont_cols_input
                    and self.target == col
                    and self.output == "classification"
                )
            )

            if self.target == col:
                a = 2

            if is_continuous:
                if verbose:
                    print(f"{col} is cont.")
                self.data[col] = pd.to_numeric(self.data[col], errors="coerce")
            else:
                if verbose:
                    print(f"{col} is cat.")
                self.data[col] = self.data[col].astype("object")

        list_discrete = self.data.select_dtypes(include=["object"]).columns.tolist()
        if self.output == "classification":
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
            "sex",
            "age",
            "birthdate",
            "questionnaire_date",
            "marital_status",
            "education",
            "cohabitant",
            "employment_type_a",
            "employment_type_b",
            "employment_type_c",
            "employment_type_d",
            "employment_type_e",
            "employment_type_f",
            "employment_type_g",
            "employment_type_h",
            "employment_type_i",
            "country_of_origin",
            "country_of_origin_specific",
            "health_status_year",
            "mi_stroke_family",
            "diabetes_family",
            "distance_work",
        ]

        self.sensitive_fields = [
            "height",
            "weight",
            "bmi",
            "high_bp",
            "sm_status",
            "sm_num_cig_grouped",
            "sm_num_cig",
            "sm_num_cigar",
            "sm_gr_tobacco",
            "sm_start",
            "sm_duration",
            "sn_status",
            "sn_quantity",
            "sn_duration",
            "sm_status_alt",
            "sn_status_alt",
            "pa_index",
            "rand36_pf",
            "rand36_rp",
            "rand36_bp",
            "rand36_gh",
            "rand36_vt",
            "rand36_sf",
            "rand36_re",
            "rand36_mh",
            "qol_d1",
            "qol_d2",
            "qol_d3",
            "qol_d4",
            "qol_d5",
            "qol_d6",
            "qol_d7",
            "qol_d8",
            "qol_d9",
            "qol_d10",
            "qol_d11",
            "qol_d12",
            "qol_d13",
            "qol_d14",
            "qol_d15",
            "qol_d16",
            "qol_d17",
        ]

    def _init_key_sensitive_fields_dummy(self):
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

    def _replace_lowercase_tag_with_nan_and_count(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, int]:
        """
        Replaces all cell values in a DataFrame with np.nan if their lowercase
        representation is exactly "tag". Returns the modified DataFrame and the count
        of replaced cells. Handles non-string types gracefully.

        Args:
            df (pd.DataFrame): The input pandas DataFrame.

        Returns:
            tuple: A tuple containing:
                - pd.DataFrame: A new DataFrame with specified values replaced by np.nan.
                - int: The total count of cells that were replaced.
        """
        # Initialize a counter for replaced cells
        replaced_count = 0

        # Use .applymap() to apply a function element-wise across the DataFrame.
        # We need to use a function that can access and modify the outer scope's counter.
        # A simple way is to make the counter non-local if using nested function,
        # or pass a mutable object (like a list or dict) to the helper if applymap
        # allowed extra args (it doesn't directly).
        # A more direct way is to calculate the boolean mask first and then sum it.

        # Create a boolean mask where True indicates a cell to be replaced
        # This checks if the cell is a string AND its lowercase is 'tag'
        # We need to apply this element-wise and handle non-string types without error.

        # A robust way is to iterate and check type, or use a method that handles non-strings.
        # .applymap() with a lambda or small function is still a good approach.
        # Let's use a list to hold the count from the inner function (workaround for applymap)
        count_list = [0]

        def _replace_value_and_count(cell_value):
            # Check if the value is a string
            if isinstance(cell_value, str):
                # If it's a string, convert to lowercase and check if it's "tag"
                if cell_value.lower() == "tag":
                    count_list[0] += 1  # Increment the counter
                    return np.nan  # Replace with np.nan
                else:
                    return cell_value  # Keep the original string value
            # If the value is not a string (e.g., number, boolean, existing NaN),
            # return the original value unchanged.
            return cell_value

        # Apply the helper function to each element of the DataFrame
        df_replaced = df.applymap(_replace_value_and_count)

        # The total count is now in count_list[0]
        total_replaced = count_list[0]

        return df_replaced, total_replaced

    def enforce_binary_columns(
        self, df: pd.DataFrame, binary_cols: list
    ) -> pd.DataFrame:
        """
        Normalize binary columns to object dtype with "0"/"1" string values.

        Rules:
        - If numeric {0,1}: map to "0"/"1" and convert to object.
        - If object and values == {"0","1"}: leave as-is.
        - If object with "-1" and one other value: map "-1"->"0", other->"1".
        - If object with "f" and "m": map "f"->"0", "m"->"1".
        - Otherwise: raise ValueError.
        """
        out = df.copy()
        for col in binary_cols:
            if col not in out.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame")

            series = out[col]
            vals = series.dropna().unique().tolist()
            val_set = set(vals)

            # Case 1: numeric dtype with only {0,1}
            if pd.api.types.is_numeric_dtype(series) and val_set.issubset({0, 1}):
                out[col] = series.map({0: "0", 1: "1"}).astype(object)
                continue

            # Case 2: object dtype already with exactly {"0","1"}
            if pd.api.types.is_object_dtype(series) and val_set == {"0", "1"}:
                out[col] = series.astype(object)
                continue

            # Case 3: object dtype with "-1" plus one other
            if (
                pd.api.types.is_object_dtype(series)
                and "-1" in val_set
                and len(val_set) == 2
            ):
                other = (val_set - {"-1"}).pop()
                out[col] = series.map({"-1": "0", other: "1"}).astype(object)
                continue

            # Case 4: object dtype with "f" and "m"
            if pd.api.types.is_object_dtype(series) and val_set == {"f", "m"}:
                out[col] = series.map({"f": "0", "m": "1"}).astype(object)
                continue

            # Otherwise unsupported
            raise ValueError(
                f"Column '{col}' has unsupported values {vals}; must be "
                "numeric {0,1}, object {'0','1'}, object with '-1' and one other, or object with 'f'/'m'."
            )

        return out

    def perform_pca(self, var_threshold=0.9, verbose=False):
        """
        1) Converts any discrete columns to str
        2) Drops any columns in self.pca_excluded_columns from the PCA
        3) Binary-encodes the remaining binary columns
        4) Runs PCA on the remaining continuous + discrete set
        5) Re-assembles a DataFrame of [ PCA components | excluded columns ]
        6) Updates self.type_columns accordingly
        """
        print(">> perform pca")

        # 1) ensure all discrete columns are str
        for col in self.data.columns:
            if self.type_columns[col] == "discrete":
                self.data[col] = self.data[col].astype(str)

        # 2) determine excluded columns: both user‐specified and any *_missing
        base_excluded = set(self.pca_excluded_columns or [])
        # detect actual missing-flag columns present in the DataFrame
        miss_flags = {
            f"{c}_missing" for c in base_excluded if f"{c}_missing" in self.data.columns
        }
        excluded = base_excluded.union(miss_flags)
        self.pca_excluded_columns = list(excluded)

        pca_input_cols = [c for c in self.data.columns if c not in excluded]
        if verbose:
            print("Excluding from PCA:", excluded)
            print("PCA on columns:", pca_input_cols)

        # 3) categorize and then remove any excluded from each list
        binary_cols, discrete_cols, cont_cols = self.categorize_columns(self.data)
        binary_cols = [c for c in binary_cols if c in pca_input_cols]
        discrete_cols = [c for c in discrete_cols if c in pca_input_cols]
        cont_cols = [c for c in cont_cols if c in pca_input_cols]
        if verbose:
            print("-> binary_cols for PCA:", binary_cols)
            print("-> discrete_cols for PCA:", discrete_cols)
            print("-> cont_cols for PCA:", cont_cols)

        # 4) binary‐encode
        self.binary_encoder = BinaryColumnEncoder(binary_cols)
        self.data = self.binary_encoder.fit_transform(self.data)

        # 5) choose/instantiate PCAProcessor
        if getattr(self, "cancer", None) == "biobank_phase3_cancer_bio_all":
            self.pca = PCAProcessor(n_components=250, plot_pca=False)
        else:
            self.pca = PCAProcessor(
                n_components=None, plot_pca=False, var_threshold=var_threshold
            )

        # 6) fit_transform PCA on the input‐cols
        X_pca = self.pca.fit_transform(
            self.data[pca_input_cols],
            cont_cols,
            discrete_cols,
            [],  # no “nested” here
            binary_cols,
        )  # returns np.array shape [n_samples, n_components]

        self.data_original_columns = (
            pca_input_cols,
            cont_cols,
            discrete_cols,
            [],
            binary_cols,
        )

        # 7) re‐attach excluded columns (in original order)
        if excluded:
            X_excl = self.data[list(excluded)].to_numpy()
            X_full = np.hstack([X_pca, X_excl])
            new_cols = [f"PC{i+1}" for i in range(X_pca.shape[1])] + list(excluded)
        else:
            X_full = X_pca
            new_cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]

        # 8) rebuild self.data & self.type_columns
        self.data = pd.DataFrame(X_full, columns=new_cols)

        # build new type_columns
        new_types = {}
        for col in new_cols:
            if col.startswith("PC"):
                new_types[col] = "continuous"
            else:
                # preserve original type
                new_types[col] = self.type_columns.get(col, "continuous")
        self.type_columns = new_types

        if verbose:
            print("Resulting DataFrame columns:", new_cols)
            print("Resulting type_columns:", self.type_columns)

    def _legacy__init__(
        self,
        cancer,
        target="cancer",
        notebook_path=None,
        is_encode=True,
        is_encode_missing=True,
        is_pca=True,
    ):
        super().__init__(notebook_path)

        dict_file = {
            "biobank_phase3_cancer_150": "nshds_cancer_150.tsv",
            "biobank_phase3_cancer_151": "nshds_cancer_151.tsv",
            "biobank_phase3_cancer_153": "nshds_cancer_153.tsv",
            "biobank_phase3_cancer_154": "nshds_cancer_154.tsv",
            "biobank_phase3_cancer_155": "nshds_cancer_155.tsv",
            "biobank_phase3_cancer_156": "nshds_cancer_156.tsv",
            "biobank_phase3_cancer_157": "nshds_cancer_157.tsv",
            "biobank_phase3_cancer_162": "nshds_cancer_162.tsv",
            "biobank_phase3_cancer_172": "nshds_cancer_172.tsv",
            "biobank_phase3_cancer_173": "nshds_cancer_173.tsv",
            "biobank_phase3_cancer_174": "nshds_cancer_174.tsv",
            "biobank_phase3_cancer_180": "nshds_cancer_180.tsv",
            "biobank_phase3_cancer_182": "nshds_cancer_182.tsv",
            "biobank_phase3_cancer_183": "nshds_cancer_183.tsv",
            "biobank_phase3_cancer_185": "nshds_cancer_185.tsv",
            "biobank_phase3_cancer_188": "nshds_cancer_188.tsv",
            "biobank_phase3_cancer_189": "nshds_cancer_189.tsv",
            "biobank_phase3_cancer_191": "nshds_cancer_191.tsv",
            "biobank_phase3_cancer_192": "nshds_cancer_192.tsv",
            "biobank_phase3_cancer_193": "nshds_cancer_193.tsv",
            "biobank_phase3_cancer_194": "nshds_cancer_194.tsv",
            "biobank_phase3_cancer_199": "nshds_cancer_199.tsv",
            "biobank_phase3_cancer_200": "nshds_cancer_200.tsv",
            "biobank_phase3_cancer_203": "nshds_cancer_203.tsv",
            "biobank_phase3_cancer_204": "nshds_cancer_204.tsv",
            "biobank_phase3_cancer_205": "nshds_cancer_205.tsv",
            "biobank_phase3_cancer_all": "nshds_cancer_all.tsv",
            "biobank_phase3_cancer_bio_all": "nshds_cancer_bio_all.tsv",
            # dummy
            "biobank_phase3_dummy_pca": "dummy.csv",
            "biobank_phase3_dummy_multi_pca": "dummy_multi.csv",
        }

        self.target = target
        self.cancer = cancer
        self.output = "classification"
        self.ref = "1800-01-01"

        # get pca excluded columns
        self.pca_excluded_columns = [
            self.target,
            "sex",
            "gender",
            "marital_status",
            "age",
            "height",
            "weight",
            "waist",
            "bmi",
        ]

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

        # update pca excluded columns
        self.pca_excluded_columns = [
            col
            for col in self.pca_excluded_columns
            if col in self.data.columns.tolist()
        ]

        self.data, replaced_count = self._replace_lowercase_tag_with_nan_and_count(
            self.data
        )
        self.data_original = self.data.copy()

        print("total count of 'tag':", replaced_count)

        self.type_columns = self._get_type_columns()
        # print("type_columns:", self.type_columns)

        if is_encode_missing:
            self._preprocess_missing()
        else:
            self._preprocess()

        # print("type_columns:", self.type_columns)

        if is_pca:
            if "dummy" in self.cancer:
                self.perform_pca(var_threshold=0.8)
            else:
                self.perform_pca(var_threshold=0.9)

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

        # print(self.data)

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

        if "dummy" in self.cancer:
            self._init_key_sensitive_fields_dummy()
        else:
            self._init_key_sensitive_fields()


class BiobankPhase3Cancer150Dataset(BiobankPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer="biobank_phase3_cancer_150",
            notebook_path=notebook_path,
            is_encode=is_encode,
        )


class BiobankPhase3Cancer151Dataset(BiobankPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer="biobank_phase3_cancer_151",
            notebook_path=notebook_path,
            is_encode=is_encode,
        )


class BiobankPhase3Cancer153Dataset(BiobankPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer="biobank_phase3_cancer_153",
            notebook_path=notebook_path,
            is_encode=is_encode,
        )


class BiobankPhase3Cancer154Dataset(BiobankPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer="biobank_phase3_cancer_154",
            notebook_path=notebook_path,
            is_encode=is_encode,
        )


class BiobankPhase3Cancer155Dataset(BiobankPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer="biobank_phase3_cancer_155",
            notebook_path=notebook_path,
            is_encode=is_encode,
        )


class BiobankPhase3Cancer156Dataset(BiobankPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer="biobank_phase3_cancer_156",
            notebook_path=notebook_path,
            is_encode=is_encode,
        )


class BiobankPhase3Cancer157Dataset(BiobankPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer="biobank_phase3_cancer_157",
            notebook_path=notebook_path,
            is_encode=is_encode,
        )


class BiobankPhase3Cancer162Dataset(BiobankPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer="biobank_phase3_cancer_162",
            notebook_path=notebook_path,
            is_encode=is_encode,
        )


class BiobankPhase3Cancer172Dataset(BiobankPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer="biobank_phase3_cancer_172",
            notebook_path=notebook_path,
            is_encode=is_encode,
        )


class BiobankPhase3Cancer173Dataset(BiobankPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer="biobank_phase3_cancer_173",
            notebook_path=notebook_path,
            is_encode=is_encode,
        )


class BiobankPhase3Cancer174Dataset(BiobankPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer="biobank_phase3_cancer_174",
            notebook_path=notebook_path,
            is_encode=is_encode,
        )


class BiobankPhase3Cancer180Dataset(BiobankPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer="biobank_phase3_cancer_180",
            notebook_path=notebook_path,
            is_encode=is_encode,
        )


class BiobankPhase3Cancer182Dataset(BiobankPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer="biobank_phase3_cancer_182",
            notebook_path=notebook_path,
            is_encode=is_encode,
        )


class BiobankPhase3Cancer183Dataset(BiobankPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer="biobank_phase3_cancer_183",
            notebook_path=notebook_path,
            is_encode=is_encode,
        )


class BiobankPhase3Cancer185Dataset(BiobankPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer="biobank_phase3_cancer_185",
            notebook_path=notebook_path,
            is_encode=is_encode,
        )


class BiobankPhase3Cancer188Dataset(BiobankPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer="biobank_phase3_cancer_188",
            notebook_path=notebook_path,
            is_encode=is_encode,
        )


class BiobankPhase3Cancer189Dataset(BiobankPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer="biobank_phase3_cancer_189",
            notebook_path=notebook_path,
            is_encode=is_encode,
        )


class BiobankPhase3Cancer191Dataset(BiobankPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer="biobank_phase3_cancer_191",
            notebook_path=notebook_path,
            is_encode=is_encode,
        )


class BiobankPhase3Cancer192Dataset(BiobankPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer="biobank_phase3_cancer_192",
            notebook_path=notebook_path,
            is_encode=is_encode,
        )


class BiobankPhase3Cancer193Dataset(BiobankPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer="biobank_phase3_cancer_193",
            notebook_path=notebook_path,
            is_encode=is_encode,
        )


class BiobankPhase3Cancer194Dataset(BiobankPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer="biobank_phase3_cancer_194",
            notebook_path=notebook_path,
            is_encode=is_encode,
        )


class BiobankPhase3Cancer199Dataset(BiobankPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer="biobank_phase3_cancer_199",
            notebook_path=notebook_path,
            is_encode=is_encode,
        )


class BiobankPhase3Cancer200Dataset(BiobankPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer="biobank_phase3_cancer_200",
            notebook_path=notebook_path,
            is_encode=is_encode,
        )


class BiobankPhase3Cancer203Dataset(BiobankPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer="biobank_phase3_cancer_203",
            notebook_path=notebook_path,
            is_encode=is_encode,
        )


class BiobankPhase3Cancer204Dataset(BiobankPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer="biobank_phase3_cancer_204",
            notebook_path=notebook_path,
            is_encode=is_encode,
        )


class BiobankPhase3Cancer205Dataset(BiobankPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(
            cancer="biobank_phase3_cancer_205",
            notebook_path=notebook_path,
            is_encode=is_encode,
        )


class BiobankPhase3CancerAllDataset(BiobankPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True, target="icd9_code"):
        super().__init__(
            cancer="biobank_phase3_cancer_all",
            notebook_path=notebook_path,
            is_encode=is_encode,
            target=target,
        )


class BiobankPhase3CancerBioAllDataset(BiobankPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True, target="icd9_code"):
        super().__init__(
            cancer="biobank_phase3_cancer_bio_all",
            notebook_path=notebook_path,
            is_encode=is_encode,
            target=target,
        )


class BiobankPhase3DummyDataset(BiobankPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True, target="cardio"):
        super().__init__(
            cancer="biobank_phase3_dummy",
            notebook_path=notebook_path,
            is_encode=is_encode,
            target=target,
            is_pca=False,
        )


class BiobankPhase3DummyPCADataset(BiobankPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True, target="cardio"):
        super().__init__(
            cancer="biobank_phase3_dummy_pca",
            notebook_path=notebook_path,
            is_encode=is_encode,
            target=target,
        )


class BiobankPhase3DummyMultiPCADataset(BiobankPhase3Dataset):
    def __init__(self, notebook_path=None, is_encode=True, target="cardio"):
        super().__init__(
            cancer="biobank_phase3_dummy_multi_pca",
            notebook_path=notebook_path,
            is_encode=is_encode,
            target=target,
        )


def main():
    s = {
        # "biobank_phase3_cancer_150": BiobankPhase3Cancer150Dataset,
        # "biobank_phase3_cancer_151": BiobankPhase3Cancer151Dataset,
        # "biobank_phase3_cancer_153": BiobankPhase3Cancer153Dataset,
        # "biobank_phase3_cancer_154": BiobankPhase3Cancer154Dataset,
        # "biobank_phase3_cancer_155": BiobankPhase3Cancer155Dataset,
        # "biobank_phase3_cancer_156": BiobankPhase3Cancer156Dataset,
        # "biobank_phase3_cancer_157": BiobankPhase3Cancer157Dataset,
        # "biobank_phase3_cancer_162": BiobankPhase3Cancer162Dataset,
        # "biobank_phase3_cancer_172": BiobankPhase3Cancer172Dataset,
        # "biobank_phase3_cancer_173": BiobankPhase3Cancer173Dataset,
        # "biobank_phase3_cancer_174": BiobankPhase3Cancer174Dataset,
        # "biobank_phase3_cancer_180": BiobankPhase3Cancer180Dataset,
        # "biobank_phase3_cancer_182": BiobankPhase3Cancer182Dataset,
        # "biobank_phase3_cancer_183": BiobankPhase3Cancer183Dataset,
        # "biobank_phase3_cancer_185": BiobankPhase3Cancer185Dataset,
        # "biobank_phase3_cancer_188": BiobankPhase3Cancer188Dataset,
        # "biobank_phase3_cancer_189": BiobankPhase3Cancer189Dataset,
        # "biobank_phase3_cancer_191": BiobankPhase3Cancer191Dataset,
        # "biobank_phase3_cancer_192": BiobankPhase3Cancer192Dataset,
        # "biobank_phase3_cancer_193": BiobankPhase3Cancer193Dataset,
        # "biobank_phase3_cancer_194": BiobankPhase3Cancer194Dataset,
        # "biobank_phase3_cancer_199": BiobankPhase3Cancer199Dataset,
        # "biobank_phase3_cancer_200": BiobankPhase3Cancer200Dataset,
        # "biobank_phase3_cancer_203": BiobankPhase3Cancer203Dataset,
        # "biobank_phase3_cancer_204": BiobankPhase3Cancer204Dataset,
        # "biobank_phase3_cancer_205": BiobankPhase3Cancer205Dataset,
        # "biobank_phase3_cancer_all": BiobankPhase3CancerAllDataset,
        # "biobank_phase3_cancer_bio_all": BiobankPhase3CancerBioAllDataset,
        "biobank_phase3_dummy": BiobankPhase3DummyDataset,
        # "biobank_phase3_dummy_pca": BiobankPhase3DummyPCADataset,
        # "biobank_phase3_dummy_multi_pca": BiobankPhase3DummyMultiPCADataset,
    }

    for k, v in s.items():
        print(k, v)
        d = globals()[DATASET_CLASSES[k]]()
        print(d.data)

        print(d.data_train)
        print(d.data_train.describe())
        print(d.data_test)
        print(d.features)
        print(d.discrete_columns)
        print(d.target)
        print(len(d.discrete_columns))
        print(len(d.continuous_columns))
        print(len(d.features))

        print(d.key_fields)
        print(d.sensitive_fields)


if __name__ == "__main__":
    main()

from engine.dataset_helper.base import *


class BiobankDataset(EvaluatedDataset):
    def __init__(self, target, path_train, notebook_path=None):
        super().__init__(notebook_path)

        self.target = target
        self.output = "classification"

        src = "database/dataset/biobank"
        folder = self._get_dataset_folder()[0]
        dst = src.replace("biobank", folder)
        self._copy_folder(src, dst)

        self.path_train = self._get_path(path_train)
        self.data = self._read_train()

        if "id" in self.data.columns:
            self.data = self.data.drop(["id"], axis=1)

        self._preprocess()

        self.type_columns = self._get_type_columns()
        self.columns = list(self.type_columns.keys())
        self.features = []
        self.discrete_columns = []

        self._split_df_save_index()
        self._setup_task()

    def _read_train(self) -> pd.DataFrame:
        return pd.read_csv(self.path_train, sep="\t", header=0)

    def _read_test(self):
        pass

    def _get_type_columns(self) -> dict:
        return {string: "discrete" for string in list(self.data.columns)}

    @abstractmethod
    def _preprocess(self, *args, **kwargs):
        raise NotImplementedError

    def _prep_ctab(self):
        self.categorical_columns = self.discrete_columns
        self.log_columns = []
        self.mixed_columns = {}
        self.general_columns = []
        self.non_categorical_columns = []
        self.integer_columns = []
        self.problem_type = {"Classification": self.target}

    @abstractmethod
    def _init_key_sensitive_fields(self, *args, **kwargs):
        raise NotImplementedError


class BiobankRecordDataset(BiobankDataset):
    def __init__(self, target, path_train, notebook_path=None, is_encode=True):
        super().__init__(target, path_train, notebook_path)

        if is_encode:
            self._encode_label()

        self._drop_duplicates()

        self._prep_ctab()
        self._prep_tabddpm()
        self._prep_tabddpm_config_toml_mlp()
        self._prep_tabddpm_config_toml_resnet()
        self._prep_tabsyn()

        self._init_key_sensitive_fields()

    @abstractmethod
    def _init_key_sensitive_fields(self, *args, **kwargs):
        raise NotImplementedError


class BiobankRecordVitalDataset(BiobankRecordDataset):
    def __init__(self, notebook_path=None, is_encode=True):
        target = "is_vital"
        folder = self._get_dataset_folder()[0]
        path_train = f"database/dataset/{folder}/20230524_biobank_data.tsv"
        super().__init__(target, path_train, notebook_path, is_encode)

    def _preprocess(self):
        print(">> preprocessing")
        self.data = self.data.fillna(-1)
        self.data = data_utils.clean_record(self.data)

        X = self.data.copy()
        for col in [
            "id",
            "vitalstatus",
            "vitalstatus_date_bin",
            "vital_span_bin",
            "life_span_bin",
            "death_date_bin",
            "dx_date_bin",
            "icd7_code",
            "icd9_code",
            "icdo2_code",
            "icdo3_code",
            "is_vital",
            "is_dead",
        ]:
            if col in self.data:
                X = X.drop([col], axis=1)

        y = self.data.pop("is_vital")

        self.data = pd.concat([X, y], axis=1)

    def _init_key_sensitive_fields(self):
        self.key_fields = [
            "subproj",  # May represent an identifiable sub-project or group
            "sex",  # Gender could be used as a quasi-identifier
            "vdc",  # If this is some geographic code, it may help in identification
            "age_bin",  # An approximate age range can help link an individual
            "llkk_txt",  # If this represents some text info, it could assist with identification
            "llkk_county_letter",  # A county-level or region-based identifier could be used to narrow down individuals
            "birth_date_bin",  # Birth year or date range could be used for re-identification
        ]

        self.sensitive_fields = [
            "bmi",  # Health-related metric
            "sm_status",  # Smoking status
            "sm_yes_no",  # Smoking binary indicator
            "unthawed_avail",  # Potentially sensitive medical-related availability
            "samp_avail",  # Sample availability could have medical implications
            "predict_cohort",  # Prediction-related info might reveal sensitive health informationdid
            "samp_date_bin",  # Sample collection date may correlate with medical events
            "qnr_date_bin",  # Questionnaire response date might link to personal history
            "qnr_span_bin",  # Same as above for questionnaires
            "is_vital",  # Death indicator is highly sensitive information
        ]


class BiobankRecordDeadDataset(BiobankRecordDataset):
    def __init__(self, notebook_path=None, is_encode=True):
        target = "is_dead"
        folder = self._get_dataset_folder()[0]
        path_train = f"database/dataset/{folder}/20230524_biobank_data.tsv"
        super().__init__(target, path_train, notebook_path, is_encode)

    def _preprocess(self):
        print(">> preprocessing")
        self.data = self.data.fillna(-1)
        self.data = data_utils.clean_record(self.data)

        # Setting up Features and Target -- is_dead
        X = self.data.copy()
        for col in [
            "id",
            "vitalstatus",
            "vitalstatus_date_bin",
            "vital_span_bin",
            "life_span_bin",
            "death_date_bin",
            "is_vital",
            "is_dead",
        ]:
            if col in X:
                X = X.drop([col], axis=1)
        y = self.data.pop("is_dead")

        self.data = pd.concat([X, y], axis=1)

    def _init_key_sensitive_fields(self):
        self.key_fields = [
            "subproj",  # May represent an identifiable sub-project or group
            "sex",  # Gender could be used as a quasi-identifier
            "vdc",  # If this is some geographic code, it may help in identification
            "age_bin",  # An approximate age range can help link an individual
            "llkk_txt",  # If this represents some text info, it could assist with identification
            "llkk_county_letter",  # A county-level or region-based identifier could be used to narrow down individuals
            "birth_date_bin",  # Birth year or date range could be used for re-identification
        ]

        self.sensitive_fields = [
            "bmi",  # Health-related metric
            "sm_status",  # Smoking status
            "sm_yes_no",  # Smoking binary indicator
            "unthawed_avail",  # Potentially sensitive medical-related availability
            "samp_avail",  # Sample availability could have medical implications
            "icd7_code",  # ICD codes related to medical diagnoses
            "icd9_code",
            "icdo2_code",
            "icdo3_code",
            "predict_cohort",  # Prediction-related info might reveal sensitive health informationdid
            "samp_date_bin",  # Sample collection date may correlate with medical events
            "qnr_date_bin",  # Questionnaire response date might link to personal history
            "dx_date_bin",  # Diagnosis date likely indicates sensitive health info
            "samp_span_bin",  # Duration or timing between events could reveal personal details
            "qnr_span_bin",  # Same as above for questionnaires
            "is_dead",  # Death indicator is highly sensitive information
        ]


class BiobankRecordICDDataset(BiobankRecordDataset):
    def __init__(self, code, notebook_path=None, is_encode=True):
        self.code = f"{code}_code"
        target = f"{code}_code"
        folder = self._get_dataset_folder()[0]
        path_train = f"database/dataset/{folder}/20230524_biobank_data.tsv"
        super().__init__(target, path_train, notebook_path, is_encode)

    def _preprocess(self):
        print(">> preprocessing")
        self.data = self.data.fillna(-1)
        self.data = data_utils.clean_record(self.data)

        # Setting up Features and Target -- is_dead
        le = LabelEncoder()
        self.data[self.code] = le.fit_transform(self.data[self.code])

        X = self.data.copy()

        if self.target == "icd7_code":
            for col in [
                "id",
                "icd7_code",
                "icd9_code",
                "icdo2_code",
            ]:
                if col in X:
                    X = X.drop([col], axis=1)
        elif self.target == "icd9_code":
            for col in [
                "id",
                "icd7_code",
                "icd9_code",
                "icdo2_code",
            ]:
                if col in X:
                    X = X.drop([col], axis=1)
        elif self.target == "icdo2_code":
            for col in [
                "id",
                "icd7_code",
                "icd9_code",
                "icdo2_code",
                "icdo3_code",
            ]:
                if col in X:
                    X = X.drop([col], axis=1)
        elif self.target == "icdo3_code":
            for col in [
                "id",
                "icd7_code",
                "icd9_code",
                "icdo2_code",
                "icdo3_code",
            ]:
                if col in X:
                    X = X.drop([col], axis=1)

        y = self.data[self.code]
        self.data = pd.concat([X, y], axis=1)


class BiobankRecordICD7Dataset(BiobankRecordICDDataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__(code="icd7", notebook_path=notebook_path, is_encode=is_encode)

    def _init_key_sensitive_fields(self):
        self.key_fields = [
            "subproj",  # May represent an identifiable sub-project or group
            "sex",  # Gender could be used as a quasi-identifier
            "vdc",  # If this is some geographic code, it may help in identification
            "age_bin",  # An approximate age range can help link an individual
            "llkk_txt",  # If this represents some text info, it could assist with identification
            "llkk_county_letter",  # A county-level or region-based identifier could be used to narrow down individuals
            "birth_date_bin",  # Birth year or date range could be used for re-identification
        ]

        self.sensitive_fields = [
            "bmi",  # Health-related metric
            "sm_status",  # Smoking status
            "sm_yes_no",  # Smoking binary indicator
            "unthawed_avail",  # Potentially sensitive medical-related availability
            "samp_avail",  # Sample availability could have medical implications
            "predict_cohort",  # Prediction-related info might reveal sensitive health informationdid
            "samp_date_bin",  # Sample collection date may correlate with medical events
            "qnr_date_bin",  # Questionnaire response date might link to personal history
            "dx_date_bin",  # Diagnosis date likely indicates sensitive health info
            "samp_span_bin",  # Duration or timing between events could reveal personal details
            "qnr_span_bin",  # Same as above for questionnaires
            "is_dead",  # Death indicator is highly sensitive information
            "vitalstatus",
            "vitalstatus_date_bin",
            "death_date_bin",
            "life_span_bin",
            "vital_span_bin",
            "is_vital",
            "icd7_code",  # ICD codes related to medical diagnoses
        ]


class BiobankRecordICD9Dataset(BiobankRecordICDDataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__("icd9", notebook_path, is_encode)

    def _init_key_sensitive_fields(self):
        self.key_fields = [
            "subproj",  # May represent an identifiable sub-project or group
            "sex",  # Gender could be used as a quasi-identifier
            "vdc",  # If this is some geographic code, it may help in identification
            "age_bin",  # An approximate age range can help link an individual
            "llkk_txt",  # If this represents some text info, it could assist with identification
            "llkk_county_letter",  # A county-level or region-based identifier could be used to narrow down individuals
            "birth_date_bin",  # Birth year or date range could be used for re-identification
        ]

        self.sensitive_fields = [
            "bmi",  # Health-related metric
            "sm_status",  # Smoking status
            "sm_yes_no",  # Smoking binary indicator
            "unthawed_avail",  # Potentially sensitive medical-related availability
            "samp_avail",  # Sample availability could have medical implications
            "predict_cohort",  # Prediction-related info might reveal sensitive health informationdid
            "samp_date_bin",  # Sample collection date may correlate with medical events
            "qnr_date_bin",  # Questionnaire response date might link to personal history
            "dx_date_bin",  # Diagnosis date likely indicates sensitive health info
            "samp_span_bin",  # Duration or timing between events could reveal personal details
            "qnr_span_bin",  # Same as above for questionnaires
            "is_dead",  # Death indicator is highly sensitive information
            "vitalstatus",
            "vitalstatus_date_bin",
            "death_date_bin",
            "life_span_bin",
            "vital_span_bin",
            "is_vital",
            "icd9_code",  # ICD codes related to medical diagnoses
            "icdo3_code",
        ]


class BiobankRecordICDO2Dataset(BiobankRecordICDDataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__("icdo2", notebook_path, is_encode)

    def _init_key_sensitive_fields(self):
        self.key_fields = [
            "subproj",  # May represent an identifiable sub-project or group
            "sex",  # Gender could be used as a quasi-identifier
            "vdc",  # If this is some geographic code, it may help in identification
            "age_bin",  # An approximate age range can help link an individual
            "llkk_txt",  # If this represents some text info, it could assist with identification
            "llkk_county_letter",  # A county-level or region-based identifier could be used to narrow down individuals
            "birth_date_bin",  # Birth year or date range could be used for re-identification
        ]

        self.sensitive_fields = [
            "bmi",  # Health-related metric
            "sm_status",  # Smoking status
            "sm_yes_no",  # Smoking binary indicator
            "unthawed_avail",  # Potentially sensitive medical-related availability
            "samp_avail",  # Sample availability could have medical implications
            "predict_cohort",  # Prediction-related info might reveal sensitive health informationdid
            "samp_date_bin",  # Sample collection date may correlate with medical events
            "qnr_date_bin",  # Questionnaire response date might link to personal history
            "dx_date_bin",  # Diagnosis date likely indicates sensitive health info
            "samp_span_bin",  # Duration or timing between events could reveal personal details
            "qnr_span_bin",  # Same as above for questionnaires
            "is_dead",  # Death indicator is highly sensitive information
            "vitalstatus",
            "vitalstatus_date_bin",
            "death_date_bin",
            "life_span_bin",
            "vital_span_bin",
            "is_vital",
            "icdo2_code",
        ]


class BiobankRecordICDO3Dataset(BiobankRecordICDDataset):
    def __init__(self, notebook_path=None, is_encode=True):
        super().__init__("icdo3", notebook_path, is_encode)

    def _init_key_sensitive_fields(self):
        self.key_fields = [
            "subproj",  # May represent an identifiable sub-project or group
            "sex",  # Gender could be used as a quasi-identifier
            "vdc",  # If this is some geographic code, it may help in identification
            "age_bin",  # An approximate age range can help link an individual
            "llkk_txt",  # If this represents some text info, it could assist with identification
            "llkk_county_letter",  # A county-level or region-based identifier could be used to narrow down individuals
            "birth_date_bin",  # Birth year or date range could be used for re-identification
        ]

        self.sensitive_fields = [
            "bmi",  # Health-related metric
            "sm_status",  # Smoking status
            "sm_yes_no",  # Smoking binary indicator
            "unthawed_avail",  # Potentially sensitive medical-related availability
            "samp_avail",  # Sample availability could have medical implications
            "predict_cohort",  # Prediction-related info might reveal sensitive health informationdid
            "samp_date_bin",  # Sample collection date may correlate with medical events
            "qnr_date_bin",  # Questionnaire response date might link to personal history
            "dx_date_bin",  # Diagnosis date likely indicates sensitive health info
            "samp_span_bin",  # Duration or timing between events could reveal personal details
            "qnr_span_bin",  # Same as above for questionnaires
            "is_dead",  # Death indicator is highly sensitive information
            "vitalstatus",
            "vitalstatus_date_bin",
            "death_date_bin",
            "life_span_bin",
            "vital_span_bin",
            "is_vital",
            "icdo3_code",
        ]


class BiobankPatientDataset(BiobankDataset):
    def __init__(self, target, path_train, notebook_path=None, is_encode=True):
        super().__init__(target, path_train, notebook_path)

        if is_encode:
            self._encode_label(merge_columns=self.merge_columns)

        self._drop_duplicates()

        self._prep_ctab()
        self._prep_tabddpm()
        self._prep_tabddpm_config_toml_mlp()
        self._prep_tabddpm_config_toml_resnet()
        self._prep_tabsyn()

    def _preprocess(self):
        print(">> preprocessing")
        self.data = self.data.fillna(-1)
        self.data = data_utils.clean_patient(self.data)

        # merge all age-columns and label encode
        self.merge_columns = []
        for col in self.data.columns:
            if "age" in col:
                self.merge_columns.append(col)
            if col in ["sample_first", "sample_last"]:
                self.merge_columns.append(col)
        # merge all age-columns and label encode


class BiobankPatientDeadDataset(BiobankPatientDataset):
    def __init__(self, notebook_path=None, is_encode=True):
        target = "dead"
        folder = self._get_dataset_folder()[0]
        path_train = f"database/dataset/{folder}/20231115_predict_data_persons.tsv"
        super().__init__(target, path_train, notebook_path, is_encode)

        self._init_key_sensitive_fields()

    def _init_key_sensitive_fields(self):
        self.key_fields = [
            "sex",
            "smoking",
            "municipality",
            "age",
        ]
        self.sensitive_fields = [
            # "sample_first",
            # "sample_last",
            # "have_sample_before_diagnosis",
            # "have_sample_after_diagnosis",
            "bmi",
            "bmi_class",
            "diagnose_1_age",
            "diagnose_2_age",
            "diagnose_3_age",
            "diagnose_4_age",
            "diagnose_5_age",
            "diagnose_6_age",
            "diagnose_8_age",
            "diagnose_9_age",
            "diagnose_10_age",
            "diagnose_13_age",
            "diagnose_14_age",
            "diagnose_16_age",
            "diagnose_17_age",
            "diagnose_18_age",
            "diagnose_19_age",
            "diagnose_20_age",
            "dead",
        ]

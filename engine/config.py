class config:

    COLORS = {
        "gray": (66 / 255, 66 / 255, 66 / 255),
        "lightblue": (179 / 255, 229 / 255, 252 / 255),
        "red": (183 / 255, 28 / 255, 28 / 255),
        "blue": (50 / 255, 116 / 255, 161 / 255),  # y
        "orange": (225 / 255, 129 / 255, 44 / 255),  # y
        "green": (59 / 255, 146 / 255, 59 / 255),  # y
        "yellow": (255 / 255, 224 / 255, 130 / 255),
        "purple": (192 / 255, 61 / 255, 62 / 255),
        "deeppurple": (81 / 255, 45 / 255, 168 / 255),  # y
        "teal": (0 / 255, 105 / 255, 92 / 255),  # y
        "skin": (243 / 255, 243 / 255, 243 / 255),
        "skintit": (128 / 255, 128 / 255, 128 / 255),
        "darkcyan": (26 / 255, 188 / 255, 156 / 255),
        "darkbrown": (97 / 255, 64 / 255, 35 / 255),
        "darkred": (139 / 255, 0 / 255, 0 / 255),
        "goldenrod": (218 / 255, 165 / 255, 32 / 255),
        "magenta": (255 / 255, 0 / 255, 255 / 255),
    }

    COLOR_DICT = {
        "gray": "#F5F5F5",
        "lightblue": "#B3E5FC",
        "red": "#E41A1C",
        "blue": "#377EB8",
        "orange": "#FF7F00",
        "green": "#4DAF4A",
        "yellow": "#FFE082",
        "purple": "#984EA3",
        "red30": "#d28a8a",
        "blue30": "#a7c5d8",
        "orange30": "#efc7a1",
        "green30": "#a5d1a5",
        "green60": "#C1DFB3",
    }

    ORIGINAL_2_NEW_DATASET_NAME = {
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

    DICT_MAPPING_MODEL = {
        "ctgan": "CTGAN",
        "ctgan*": "CTGAN*",
        "copulagan": "CopulaGAN",
        "copulagan*": "CopulaGAN*",
        "dpcgans": "DP-CGANS",
        "dpcgans*": "DP-CGANS*",
        "ctab": "CTAB-GAN",
        "tvae": "TVAE",
        "tabddpm-mlp": "TabDDPM-MLP",
        "tabddpm-resnet": "TabDDPM-ResNet",
        "tabddpm_mlp": "TabDDPM-MLP",
        "tabddpm_resnet": "TabDDPM-ResNet",
        "tabsyn": "TabSyn",
    }

    DICT_MAPPING_EVALUATION = {
        "statistics": "Statistics",
        "ml": "ML",
        "dp": "DP",
        "overall": "Overall",
    }

    DICT_MAPPING_DATASET = {
        "abalone": "Abalone",
        "house": "House",
        "news": "News",
        "churn2": "Churn2",
        "buddy": "Buddy",
        "insurance": "Insurance",
        "king": "King",
        "adult": "Adult",
        "cardio": "Cardio",
        "wilt": "Wilt",
        "diabetes_openml": "Diabetes-ML",
        "gesture": "Gesture",
        "california": "California",
        "house_16h": "House-16h",
        "diabetesbalanced": "Diabetes Bal.",
        "diabetes": "Diabetes",
        "mnist12": "MNIST12",
        "credit": "Credit",
        "higgs-small": "Higgs-Small",
        "miniboone": "Miniboone",
    }

    DATASET_CLASSES = {
        "adult": "AdultDataset",
        "covertype": "CovertypeDataset",
        "credit": "CreditDataset",
        "intrusion": "IntrusionDataset",
        "mnist12": "MNIST12x12Dataset",
        "mnist28": "MNIST28x28Dataset",
        "diabetes": "DiabetesDataset",
        "diabetesbalanced": "DiabetesBalancedDataset",
        "news": "NewsDataset",
        "house": "HouseDataset",
        "abalone": "AbaloneDataset",
        "buddy": "BuddyDataset",
        "california": "CaliforniaDataset",
        "churn2": "Churn2Dataset",
        "diabetes_openml": "DiabetesOpenMLDataset",
        "cardio": "CardioDataset",
        "fb-comments": "FBCommentsDataset",
        "gesture": "GestureDataset",
        "higgs-small": "HiggsSmallDataset",
        "house_16h": "House16HDataset",
        "insurance": "InsuranceDataset",
        "king": "KingDataset",
        "miniboone": "MiniBooneDataset",
        "wilt": "WiltDataset",
        # biobank phase 1
        "biobank_record_vital": "BiobankRecordVitalDataset",
        "biobank_record_dead": "BiobankRecordDeadDataset",
        "biobank_record_icd7": "BiobankRecordICD7Dataset",
        "biobank_record_icd9": "BiobankRecordICD9Dataset",
        "biobank_record_icdo2": "BiobankRecordICDO2Dataset",
        "biobank_record_icdo3": "BiobankRecordICDO3Dataset",
        "biobank_patient_dead": "BiobankPatientDeadDataset",
        # biobank sensitive phase 2
        "biobank_sen_prostate": "BiobankSensitiveProstateDataset",
        "biobank_sen_breast": "BiobankSensitiveBreastDataset",
        "biobank_sen_colorectal": "BiobankSensitiveColorectalDataset",
        "biobank_sen_uroandkid": "BiobankSensitiveUrothelialAndKidneyDataset",
        "biobank_sen_lung": "BiobankSensitiveLungDataset",
        "biobank_sen_pancreatic": "BiobankSensitivePancreaticDataset",
        "biobank_sen_haematological": "BiobankSensitiveHaematologicalDataset",
        "biobank_sen_meta": "BiobankSensitiveMetabolomicsDataset",
        # biobank sensitive phase 3
        "biobank_phase3_cancer_150": "BiobankPhase3Cancer150Dataset",
        "biobank_phase3_cancer_151": "BiobankPhase3Cancer151Dataset",
        "biobank_phase3_cancer_153": "BiobankPhase3Cancer153Dataset",
        "biobank_phase3_cancer_154": "BiobankPhase3Cancer154Dataset",
        "biobank_phase3_cancer_155": "BiobankPhase3Cancer155Dataset",
        "biobank_phase3_cancer_156": "BiobankPhase3Cancer156Dataset",
        "biobank_phase3_cancer_157": "BiobankPhase3Cancer157Dataset",
        "biobank_phase3_cancer_162": "BiobankPhase3Cancer162Dataset",
        "biobank_phase3_cancer_172": "BiobankPhase3Cancer172Dataset",
        "biobank_phase3_cancer_173": "BiobankPhase3Cancer173Dataset",
        "biobank_phase3_cancer_174": "BiobankPhase3Cancer174Dataset",
        "biobank_phase3_cancer_180": "BiobankPhase3Cancer180Dataset",
        "biobank_phase3_cancer_182": "BiobankPhase3Cancer182Dataset",
        "biobank_phase3_cancer_183": "BiobankPhase3Cancer183Dataset",
        "biobank_phase3_cancer_185": "BiobankPhase3Cancer185Dataset",
        "biobank_phase3_cancer_188": "BiobankPhase3Cancer188Dataset",
        "biobank_phase3_cancer_189": "BiobankPhase3Cancer189Dataset",
        "biobank_phase3_cancer_191": "BiobankPhase3Cancer191Dataset",
        "biobank_phase3_cancer_192": "BiobankPhase3Cancer192Dataset",
        "biobank_phase3_cancer_193": "BiobankPhase3Cancer193Dataset",
        "biobank_phase3_cancer_194": "BiobankPhase3Cancer194Dataset",
        "biobank_phase3_cancer_199": "BiobankPhase3Cancer199Dataset",
        "biobank_phase3_cancer_200": "BiobankPhase3Cancer200Dataset",
        "biobank_phase3_cancer_203": "BiobankPhase3Cancer203Dataset",
        "biobank_phase3_cancer_204": "BiobankPhase3Cancer204Dataset",
        "biobank_phase3_cancer_205": "BiobankPhase3Cancer205Dataset",
        "biobank_phase3_cancer_all": "BiobankPhase3CancerAllDataset",
        "biobank_phase3_cancer_bio_all": "BiobankPhase3CancerBioAllDataset",
        # biobank phase 3 dummy
        "biobank_phase3_dummy": "BiobankPhase3DummyDataset",
        "biobank_phase3_dummy_missing": "BiobankPhase3DummyMissingDataset",
        "biobank_phase3_dummy_small": "BiobankPhase3DummySmallDataset",
        "biobank_phase3_dummy_missing_small": "BiobankPhase3DummyMissingSmallDataset",
        "biobank_phase3_dummy_pca": "BiobankPhase3DummyPCADataset",
        "biobank_phase3_dummy_multi_pca": "BiobankPhase3DummyMultiPCADataset",
    }

    DICT_MAPPING_METRICS = {  # higher/lower is better
        # stat. evaluation
        "kl_divergence_discrete": "lower",
        "kl_divergence_continuous": "lower",
        "chisquare_discrete": "lower",
        "kolmogorov_smirnov_continuous": "higher",
        "cramer_discrete": "higher",
        "pearson_continuous": "higher",
        "dwp_discrete": "lower",
        "dwp_continuous": "lower",
        "mse": "lower",
        "mae": "lower",
        # dp evaluation
        "dp_categorical_zero_cap": "higher",  # higher is better
        "dp_categorical_generalized_cap": "higher",  # higher is better
        "dp_dcr_rf": "higher",  # higher is better
        "dp_nndr_rf": "higher",  # higher is better
        "dp_single_out_": "lower",  # all dp single out smaller is better -- different n_cols
        "dp_linkability_": "lower",  # all dp linkability smaller is better -- different n_neighbors
        "dp_inference_": "lower",  # all dp inference smaller is better -- different secret
        "dp_k_anonymization_synthetic": "higher",  # higher is better
        "dp_k_anonymization_safe": "higher",  # higher is better
        "dp_l_diversity_synthetic": "higher",  # higher is better
        "dp_l_diversity_safe": "higher",  # higher is better
        "dp_k_map": "higher",  # higher is better
        "dp_delta_presence": "lower",  # smaller is better
        "dp_re_identification_score": "lower",  # smaller is better
        "dp_domias_mia_accuracy": "lower",  # smaller is better
        "dp_domias_mia_auc": "lower",  # smaller is better
    }

    DICT_DATASETS = {
        "small": {
            "dataset": [
                "abalone",
                "adult",
                "buddy",
                "california",
                "churn2",
                "diabetes_openml",
                "diabetesbalanced",
                "gesture",
                "house",
                "house_16h",
                "insurance",
                "king",
                "news",
                "wilt",
                "biobank_record_vital",
                "biobank_record_dead",
                "biobank_record_icd7",
                "biobank_record_icd9",
                "biobank_record_icdo2",
                "biobank_record_icdo3",
                "biobank_patient_dead",
                "biobank_sen_prostate",
                "biobank_sen_breast",
                "biobank_sen_colorectal",
                "biobank_sen_uroandkid",
                "biobank_sen_lung",
                "biobank_sen_pancreatic",
                "biobank_sen_haematological",
                "biobank_sen_meta",
                "biobank_phase3_cancer_150",
                "biobank_phase3_cancer_151",
                "biobank_phase3_cancer_155",
                "biobank_phase3_cancer_156",
                "biobank_phase3_cancer_157",
                "biobank_phase3_cancer_180",
                "biobank_phase3_cancer_182",
                "biobank_phase3_cancer_183",
                "biobank_phase3_cancer_189",
                "biobank_phase3_cancer_191",
                "biobank_phase3_cancer_192",
                "biobank_phase3_cancer_193",
                "biobank_phase3_cancer_194",
                "biobank_phase3_cancer_199",
                "biobank_phase3_cancer_203",
                "biobank_phase3_cancer_204",
                "biobank_phase3_cancer_205",
            ],
            "epochs_max": 2000,
            "max_trials": 30,
        },
        "medium": {
            "dataset": [
                "cardio",
                "diabetes",
                "mnist12",
                "credit",
                "higgs-small",
                "miniboone",
                "fb-comments",
            ],
            "epochs_max": 2000,
            "max_trials": 30,
        },
        "large": {
            "dataset": [
                "covertype",
                "intrusion",
                "mnist28",
            ],
            "epochs_max": 1000,
            "max_trials": 20,
        },
    }

    DICT_STRING_DATASETS = {
        "abalone": "Abalone",
        "adult": "Adult",
        "buddy": "Buddy",
        "california": "California",
        "cardio": "Cardio",
        "churn2": "Churn2",
        "covertype": "Covertype",
        "credit": "Credit",
        "diabetes": "Diabetes",
        "diabetes_openml": "Diabetes-ML",
        "diabetesbalanced": "Diabetes Bal.",
        "fb-comments": "Fb-Comments",
        "gesture": "Gesture",
        "higgs-small": "Higgs-Small",
        "house": "House",
        "house_16h": "House-16h",
        "insurance": "Insurance",
        "intrusion": "Intrusion",
        "king": "King",
        "miniboone": "Miniboone",
        "mnist12": "MNIST12",
        "mnist28": "MNIST28",
        "news": "News",
        "wilt": "Wilt",
    }

    _DICT_COLOR = {
        "copulagan_condvec_0": "red",
        "copulagan_condvec_1": "green",
        "ctab_condvec_1": "blue",
        "ctgan_condvec_0": "purple",
        "ctgan_condvec_1": "teal",
        "dpcgans_condvec_0": "indigo",
        "dpcgans_condvec_1": "darkorange",
        "tvae_condvec_1": "black",
    }

    DICT_COLOR = {
        "ctgan": "red",
        "copulagan": "blue",
        "ctab": "green",
        "dpcgans": "purple",
        "tvae": "black",
    }

    DICT_METRIC_MAPPING = {
        "kl_divergence_discrete": "transform",
        "kl_divergence_continuous": "transform",
        "chisquare_discrete": "transform",
        "kolmogorov_smirnov_continuous": "transform",
        # "cramer_discrete": "keep",
        # "pearson_continuous": "keep",
        "cramer_discrete": "transform",
        "pearson_continuous": "transform",
        "dwp_discrete": "transform",
        "dwp_continuous": "transform",
    }

    LIST_OPTIMAL = {
        "none": {
            "losscorr_0-lossdwp_0-condvec_1": "fake_01850.csv",
            "losscorr_1-lossdwp_1-condvec_0": "fake_01700.csv",
        },
        "sigma_0.0001": {
            "losscorr_0-lossdwp_0-condvec_1": "fake_01850.csv",
            "losscorr_1-lossdwp_1-condvec_0": "fake_01700.csv",
        },
        "sigma_0.001": {
            "losscorr_0-lossdwp_0-condvec_1": "fake_02600.csv",
            "losscorr_1-lossdwp_1-condvec_0": "fake_01700.csv",
        },
        "sigma_0.01": {
            "losscorr_0-lossdwp_0-condvec_1": "fake_02150.csv",
            "losscorr_1-lossdwp_1-condvec_0": "fake_01850.csv",
        },
        "sigma_0.1": {
            "losscorr_0-lossdwp_0-condvec_1": "fake_02250.csv",
            "losscorr_1-lossdwp_1-condvec_0": "fake_01950.csv",
        },
        "sigma_1.0": {
            "losscorr_0-lossdwp_0-condvec_1": "fake_02350.csv",
            "losscorr_1-lossdwp_1-condvec_0": "fake_01650.csv",
        },
        "sigma_10.0": {
            "losscorr_0-lossdwp_0-condvec_1": "fake_02600.csv",
            "losscorr_1-lossdwp_1-condvec_0": "fake_03000.csv",
        },
    }


class config_last:
    LIST_OPTIMAL = {
        "none": {
            "losscorr_0-lossdwp_0-condvec_1": "fake_01850.csv",
            "losscorr_1-lossdwp_1-condvec_0": "fake_01700.csv",
        },
        "sigma_0.0001": {
            "losscorr_0-lossdwp_0-condvec_1": "fake_01850.csv",
            "losscorr_1-lossdwp_1-condvec_0": "fake_01700.csv",
        },
        "sigma_0.001": {
            "losscorr_0-lossdwp_0-condvec_1": "fake_02600.csv",
            "losscorr_1-lossdwp_1-condvec_0": "fake_01700.csv",
        },
        "sigma_0.01": {
            "losscorr_0-lossdwp_0-condvec_1": "fake_02150.csv",
            "losscorr_1-lossdwp_1-condvec_0": "fake_01850.csv",
        },
        "sigma_0.1": {
            "losscorr_0-lossdwp_0-condvec_1": "fake_02250.csv",
            "losscorr_1-lossdwp_1-condvec_0": "fake_01950.csv",
        },
        "sigma_1.0": {
            "losscorr_0-lossdwp_0-condvec_1": "fake_02350.csv",
            "losscorr_1-lossdwp_1-condvec_0": "fake_01650.csv",
        },
        "sigma_10.0": {
            "losscorr_0-lossdwp_0-condvec_1": "fake_02600.csv",
            "losscorr_1-lossdwp_1-condvec_0": "fake_03000.csv",
        },
    }

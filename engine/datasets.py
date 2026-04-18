from engine.dataset_helper.base import DATASET_CLASSES
from engine.dataset_helper.public import *
from engine.dataset_helper.biobank_phase1 import *
from engine.dataset_helper.biobank_phase2 import *
from engine.dataset_helper.biobank_phase3 import *

# from engine.dataset_helper.dummy_phase3 import *


def get_dataset(dataset, arch=None, is_encode=True, notebook_path=None):
    # Mapping dataset names to their corresponding classes

    if dataset not in DATASET_CLASSES:
        raise NotImplementedError(f"The dataset '{dataset}' is not implemented.")

    # Instantiate the dataset object
    d = globals()[DATASET_CLASSES[dataset]]
    try:
        D = d(is_encode=is_encode, notebook_path=notebook_path)
    except:
        D = d(notebook_path=notebook_path)

    # Special case handling
    if dataset == "intrusion" and arch == "copulagan":
        D._drop_duplicates()

    return D


def main():
    s = {
        # "biobank_patient_dead": BiobankPatientDeadDataset,
        # "biobank_record_dead": BiobankRecordDeadDataset,
        # "biobank_record_vital": BiobankRecordVitalDataset,
        # "biobank_record_icd7": BiobankRecordICD7Dataset,
        # "biobank_record_icd9": BiobankRecordICD9Dataset,
        # "biobank_record_icdo2": BiobankRecordICDO2Dataset,
        # "biobank_record_icdo3": BiobankRecordICDO3Dataset,
        # "adult": AdultDataset,
        # "abalone": AbaloneDataset,
        # "diabetesbalanced": DiabetesBalancedDataset,
        # "biobank_sen_prostate": BiobankSensitiveProstateDataset,
        # "biobank_sen_breast": BiobankSensitiveBreastDataset,
        # "biobank_sen_colorectal": BiobankSensitiveColorectalDataset,
        # "biobank_sen_uroandkid": BiobankSensitiveUrothelialAndKidneyDataset,
        # "biobank_sen_lung": BiobankSensitiveLungDataset,
        # "biobank_sen_pancreatic": BiobankSensitivePancreaticDataset,
        # "biobank_sen_haematological": BiobankSensitiveHaematologicalDataset,
        # "biobank_sen_meta": BiobankSensitiveMetabolomicsDataset,
        # -------------------------
        # phase 3
        # -------------------------
        "biobank_phase3_cancer_bio_all": BiobankPhase3CancerBioAllDataset,
        "biobank_phase3_cancer_all": BiobankPhase3CancerAllDataset,
        # "biobank_phase3_cancer_185": BiobankPhase3Cancer185Dataset,
        # "biobank_phase3_cancer_174": BiobankPhase3Cancer174Dataset,
        # "biobank_phase3_cancer_153": BiobankPhase3Cancer153Dataset,
        # "biobank_phase3_cancer_162": BiobankPhase3Cancer162Dataset,
        # "biobank_phase3_cancer_188": BiobankPhase3Cancer188Dataset,
        # "biobank_phase3_cancer_172": BiobankPhase3Cancer172Dataset,
        # "biobank_phase3_cancer_154": BiobankPhase3Cancer154Dataset,
        # "biobank_phase3_cancer_173": BiobankPhase3Cancer173Dataset,
        # "biobank_phase3_cancer_200": BiobankPhase3Cancer200Dataset,
        # -------------------------
        # phase 3 later
        # -------------------------
        # "biobank_phase3_cancer_150": BiobankPhase3Cancer150Dataset,
        # "biobank_phase3_cancer_151": BiobankPhase3Cancer151Dataset,
        # "biobank_phase3_cancer_155": BiobankPhase3Cancer155Dataset,
        # "biobank_phase3_cancer_156": BiobankPhase3Cancer156Dataset,
        # "biobank_phase3_cancer_157": BiobankPhase3Cancer157Dataset,
        # "biobank_phase3_cancer_180": BiobankPhase3Cancer180Dataset,
        # "biobank_phase3_cancer_182": BiobankPhase3Cancer182Dataset,
        # "biobank_phase3_cancer_183": BiobankPhase3Cancer183Dataset,
        # "biobank_phase3_cancer_189": BiobankPhase3Cancer189Dataset,
        # "biobank_phase3_cancer_191": BiobankPhase3Cancer191Dataset,
        # "biobank_phase3_cancer_192": BiobankPhase3Cancer192Dataset,
        # "biobank_phase3_cancer_193": BiobankPhase3Cancer193Dataset,
        # "biobank_phase3_cancer_194": BiobankPhase3Cancer194Dataset,
        # "biobank_phase3_cancer_199": BiobankPhase3Cancer199Dataset,
        # "biobank_phase3_cancer_203": BiobankPhase3Cancer203Dataset,
        # "biobank_phase3_cancer_204": BiobankPhase3Cancer204Dataset,
        # "biobank_phase3_cancer_205": BiobankPhase3Cancer205Dataset,
        # -------------------------
        # phase 3 dummy pca
        # -------------------------
        # "biobank_phase3_dummy": BiobankPhase3DummyDataset,
        # "biobank_phase3_dummy_missing": BiobankPhase3DummyMissingDataset,
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
        print(d.key_fields)
        print(d.sensitive_fields)

        print("len(d.discrete_columns):", len(d.discrete_columns))
        print("len(d.continuous_columns):", len(d.continuous_columns))
        print("len(d.features):", len(d.features))

        # correlation_matrix = d.data.corr()
        # path = os.path.join(d.notebook_path, "biobank_phase3_cancer_200_corr.tsv")
        # correlation_matrix.to_csv(path, sep="\t", index=True)


if __name__ == "__main__":
    main()

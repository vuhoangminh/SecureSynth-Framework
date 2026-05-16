import os
import pandas as pd
from pathlib import Path

from engine.dataset_helper.base import EvaluatedDataset
from engine.config_loader import load_config


class GenericDataset(EvaluatedDataset):
    """Config-driven dataset loader for any tabular CSV/TSV/parquet."""

    def __init__(self, config_path: str, is_encode: bool = True, notebook_path=None):
        self._dataset_name = Path(config_path).stem
        self._cfg = load_config(config_path)

        super().__init__(notebook_path)

        self.target = self._cfg.columns.target
        self.output = self._cfg.columns.task
        self.features = []
        self.discrete_columns = []
        self.key_fields = self._cfg.attributes.key
        self.sensitive_fields = self._cfg.attributes.sensitive

        self.path_train = self._get_path(self._cfg.data.path)

        self.data = self._read_train()
        self.data_train = self.data.copy()
        self.type_columns = self._get_type_columns()

        os.makedirs(os.path.dirname(self.pkl_path), exist_ok=True)
        self._split_df_save_index()
        self._setup_task()
        self.columns = list(self.type_columns.keys())

        if is_encode:
            self._encode_label()

        self._prep_ctab()
        self._prep_tabddpm()
        try:
            self._prep_tabddpm_config_toml_mlp()
            self._prep_tabddpm_config_toml_resnet()
        except Exception:
            pass
        self._prep_tabsyn()

    def _get_dataset_folder(self):
        return [self._dataset_name]

    def _get_class_name(self):
        return self._dataset_name

    def _get_base_path(self):
        # TabSyn/TabDDPM expect their artefacts under database/dataset/{name}/,
        # not under the raw data directory (data/).
        return self._get_path(f"database/prepared/{self._dataset_name}")

    def _read_train(self) -> pd.DataFrame:
        cfg = self._cfg.data
        if cfg.format == "parquet":
            df = pd.read_parquet(self.path_train)
        elif cfg.format == "tsv":
            df = pd.read_csv(self.path_train, sep="\t")
        else:
            df = pd.read_csv(self.path_train, sep=cfg.separator)
        if cfg.drop_columns:
            df = df.drop(columns=[c for c in cfg.drop_columns if c in df.columns])
        return df

    def _read_test(self) -> pd.DataFrame:
        pass

    def _get_type_columns(self) -> dict:
        cfg = self._cfg.columns
        explicit_cont = set(cfg.continuous)
        explicit_disc = set(cfg.discrete)

        type_cols = {}
        for col in self.data_train.columns:
            if col in explicit_cont:
                type_cols[col] = "continuous"
            elif col in explicit_disc:
                type_cols[col] = "discrete"
            else:
                # auto-detect: n_unique > 15 → continuous, else discrete
                n_unique = self.data_train[col].dropna().nunique()
                type_cols[col] = "continuous" if n_unique > 15 else "discrete"
        return type_cols

    def postprocess_synthetic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply TOML [postprocessing].constraints to a decoded synthetic DataFrame.

        Each constraint is a pandas df.query() expression. Rows that violate any
        constraint are dropped. Logs dropped counts per constraint.
        """
        constraints = self._cfg.postprocessing.constraints
        if not constraints:
            return df
        n_before = len(df)
        for constraint in constraints:
            n_pre = len(df)
            df = df.query(constraint)
            n_dropped = n_pre - len(df)
            if n_dropped:
                print(f"postprocess: dropped {n_dropped} rows violating '{constraint}'")
        n_total_dropped = n_before - len(df)
        if n_total_dropped:
            print(f"postprocess: {n_before} → {len(df)} rows ({n_total_dropped} dropped total)")
        return df.reset_index(drop=True)

    def _prep_ctab(self):
        # CTAB-GAN params are normally loaded from database/dataset/ctab_columns.json
        # which only covers biobank datasets. Set safe defaults for generic datasets.
        self.general_columns = []
        self.non_categorical_columns = []
        self.log_columns = []
        self.integer_columns = []
        self.problem_type = {self.target: self.output}

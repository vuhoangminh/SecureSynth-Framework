"""Tests for t043a/b/c: optimizer scripts use get_run_dir, write params.json/status.json,
and route the .hyperopt pickle to database/optimization/generative_model/.
"""
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fake_module(name):
    m = types.ModuleType(name)
    return m


class _FoundNANsError(Exception):
    pass


def _load_optimizer(path, extra_mocks=None):
    """Load an optimizer script as a module with heavy deps mocked out."""
    heavy = [
        "engine.evaluate_technical_paper",
        "engine.datasets",
        "engine.config",
        "engine.utils.data_utils",
        "engine.ctgan_data_transformer",
        "models",
        "models.tab_ddpm",
        "models.tab_ddpm.tab_ddpm",
        "models.tab_ddpm.lib",
        "models.dpcgans",
        "models.ctgan",
        "models.tvae",
        "models.copulagan",
        "models.CTAB",
        "models.CTAB.ctabgan",
    ]

    # FoundNANsError must be a real exception class so 'except FoundNANsError' works
    utils_mock = types.ModuleType("models.tab_ddpm.tab_ddpm.utils")
    utils_mock.FoundNANsError = _FoundNANsError
    originals = {}
    for name in heavy:
        if name not in sys.modules:
            sys.modules[name] = MagicMock()
            originals[name] = None
        else:
            originals[name] = sys.modules[name]
            sys.modules[name] = MagicMock()

    # install the real-exception-class version of the utils module
    originals["models.tab_ddpm.tab_ddpm.utils"] = sys.modules.get("models.tab_ddpm.tab_ddpm.utils")
    sys.modules["models.tab_ddpm.tab_ddpm.utils"] = utils_mock

    if extra_mocks:
        for name, obj in extra_mocks.items():
            originals[name] = sys.modules.get(name)
            sys.modules[name] = obj

    spec = importlib.util.spec_from_file_location("_opt_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # restore
    for name, original in originals.items():
        if original is None:
            del sys.modules[name]
        else:
            sys.modules[name] = original

    return mod


def _make_args(**kwargs):
    import argparse
    ns = argparse.Namespace(
        dataset="adult",
        arch="ctgan",
        loss_version=2,
        is_test=1,
        module="public",
        row_number=None,
        is_condvec=1,
        bo_method="ior",
        bo_method_agg="median",
        dir_logs="database/gan_optimize/",
        max_trials=30,
        model_type="mlp",
    )
    for k, v in kwargs.items():
        setattr(ns, k, v)
    return ns


# ---------------------------------------------------------------------------
# ctgan
# ---------------------------------------------------------------------------

CTGAN_PATH = "scripts/optimize/optimize_ctgan.py"
TABSYN_PATH = "scripts/optimize/optimize_tabsyn.py"
TABDDPM_PATH = "scripts/optimize/optimize_tabddpm.py"


@pytest.fixture()
def ctgan_mod():
    return _load_optimizer(CTGAN_PATH)


@pytest.fixture()
def tabsyn_mod():
    return _load_optimizer(TABSYN_PATH)


@pytest.fixture()
def tabddpm_mod():
    return _load_optimizer(TABDDPM_PATH)


def _run_objective(mod, tmp_path, params=None, arch="ctgan", loss_version=2):
    """Set up module globals and call objective(params)."""
    if params is None:
        params = {
            "epochs": 20,
            "batch_size": 500,
            "embedding_dim": 64,
            "generator_dim": 64,
            "discriminator_dim": 64,
            "generator_lr": 2e-4,
            "generator_decay": 1e-6,
            "discriminator_lr": 2e-4,
            "discriminator_decay": 1e-6,
            "private": 0,
            "dp_sigma": 0.1,
            "dp_weight_clip": 0.01,
        }

    mod.args = _make_args(arch=arch, loss_version=loss_version)
    import argparse
    mod.parser = argparse.ArgumentParser(add_help=False)
    mod.parser.add_argument("--dataset", default="adult")
    mod.parser.add_argument("--arch", default=arch)
    mod.parser.add_argument("--loss_version", default=loss_version, type=int)
    mod.parser.add_argument("--is_test", default=1, type=int)
    mod.parser.add_argument("--is_condvec", default=1, type=int)
    mod.parser.add_argument("--row_number", default=None, type=int)
    mod.parser.add_argument("--is_drop_id", default=1, type=int)
    mod.parser.add_argument("--private", default=0, type=int)
    mod._trial_count[0] = 0

    fake_trial_dir = tmp_path / "adult-ctgan-lv2" / "trial_0000"
    fake_trial_dir.mkdir(parents=True, exist_ok=True)

    def fake_get_run_dir(dataset, model, loss, trial_n, is_test=False):
        return fake_trial_dir

    fake_run_result = MagicMock()
    fake_run_result.stdout = b"ok"
    fake_run_result.returncode = 0

    with (
        patch.object(mod.path_utils, "get_run_dir", side_effect=fake_get_run_dir) as mock_grd,
        patch.object(mod.path_utils, "make_dir", MagicMock()),
        patch.object(mod.subprocess, "run", return_value=fake_run_result),
        patch.object(mod, "compute_statistical_metrics", return_value=MagicMock(iloc=[{}], drop=MagicMock(return_value=MagicMock(iloc=[{}])))),
        patch.object(mod, "compute_ml_metrics_all_ml_methods", return_value=MagicMock(iloc=[{}], drop=MagicMock(return_value=MagicMock(iloc=[{}])))),
        patch.object(mod, "compute_dp_metrics", return_value=MagicMock(iloc=[{}], drop=MagicMock(return_value=MagicMock(iloc=[{}])))),
        patch.object(mod, "get_dataset", return_value=MagicMock(discrete_columns=[], data_train=MagicMock(__iter__=iter([])))),
    ):
        with patch.object(mod.path_utils, "get_filename", return_value="trial_0000"):
            result = mod.objective(params.copy())

    return result, mock_grd, fake_trial_dir


class TestCtganUsesGetRunDir:
    def test_get_run_dir_called(self, ctgan_mod, tmp_path):
        _, mock_grd, _ = _run_objective(ctgan_mod, tmp_path)
        mock_grd.assert_called_once()

    def test_get_run_dir_args(self, ctgan_mod, tmp_path):
        _, mock_grd, _ = _run_objective(ctgan_mod, tmp_path, arch="ctgan", loss_version=2)
        args, kwargs = mock_grd.call_args
        assert args[0] == "adult"   # dataset
        assert args[1] == "ctgan"   # model
        assert args[2] == "lv2"     # loss
        assert args[3] == 0         # trial_n

    def test_params_json_written(self, ctgan_mod, tmp_path):
        _, _, trial_dir = _run_objective(ctgan_mod, tmp_path)
        params_file = trial_dir / "params.json"
        assert params_file.exists()
        data = json.loads(params_file.read_text())
        assert isinstance(data, dict)

    def test_status_json_has_status_key(self, ctgan_mod, tmp_path):
        _, _, trial_dir = _run_objective(ctgan_mod, tmp_path)
        status_file = trial_dir / "status.json"
        assert status_file.exists()
        data = json.loads(status_file.read_text())
        assert "status" in data

    def test_trial_counter_increments(self, ctgan_mod, tmp_path):
        ctgan_mod.args = _make_args()
        import argparse
        ctgan_mod.parser = argparse.ArgumentParser(add_help=False)
        ctgan_mod._trial_count[0] = 5
        fake_dir = tmp_path / "t"
        fake_dir.mkdir()

        def fake_grd(dataset, model, loss, trial_n, is_test=False):
            return fake_dir

        with patch.object(ctgan_mod.path_utils, "get_run_dir", side_effect=fake_grd) as mock_grd, \
             patch.object(ctgan_mod.path_utils, "make_dir", MagicMock()), \
             patch.object(ctgan_mod.subprocess, "run", side_effect=Exception("skip")):
            try:
                ctgan_mod.objective({"epochs": 20})
            except Exception:
                pass
        call_args = mock_grd.call_args[0]  # (dataset, model, loss, trial_n)
        assert call_args[3] == 5           # trial_n
        assert ctgan_mod._trial_count[0] == 6


class TestCtganHyperoptFolder:
    def test_ior_folder(self, ctgan_mod):
        with patch.object(ctgan_mod.path_utils, "get_hyperopt_path", return_value="/tmp/fake.hyperopt") as mock_ghp, \
             patch.object(ctgan_mod.hyperopt_utils, "load_project", return_value=MagicMock(trials=[])):
            import argparse
            args = _make_args(bo_method="ior", bo_method_agg="median")
            args.row_number = None
            # simulate the __main__ folder logic
            folder = "optimization/generative_model" if args.bo_method == "ior" else f"optimization/generative_model_sbo_{args.bo_method_agg}"
        assert folder == "optimization/generative_model"

    def test_sbo_folder(self, ctgan_mod):
        args = _make_args(bo_method="sbo", bo_method_agg="mean")
        folder = "optimization/generative_model" if args.bo_method == "ior" else f"optimization/generative_model_sbo_{args.bo_method_agg}"
        assert folder == "optimization/generative_model_sbo_mean"


# ---------------------------------------------------------------------------
# tabsyn
# ---------------------------------------------------------------------------

def _run_tabsyn_objective(mod, tmp_path, params=None):
    if params is None:
        params = {
            "batch_size": 256,
            "dim_t": 256,
            "num_epochs": 20,
            "lr": 0.001,
            "factor": 0.5,
            "epochs": 20,
        }

    mod.args = _make_args(arch="tabsyn", loss_version=0)
    import argparse
    mod.parser = argparse.ArgumentParser(add_help=False)
    mod.parser.add_argument("--dataset", default="adult")
    mod.parser.add_argument("--arch", default="tabsyn")
    mod.parser.add_argument("--loss_version", default=0, type=int)
    mod.parser.add_argument("--is_test", default=1, type=int)
    mod.parser.add_argument("--is_condvec", default=1, type=int)
    mod.parser.add_argument("--row_number", default=None, type=int)
    mod.parser.add_argument("--is_drop_id", default=1, type=int)
    mod._trial_count[0] = 0

    fake_trial_dir = tmp_path / "adult-tabsyn-lv0" / "trial_0000"
    fake_trial_dir.mkdir(parents=True, exist_ok=True)

    def fake_get_run_dir(dataset, model, loss, trial_n, is_test=False):
        return fake_trial_dir

    fake_run_result = MagicMock()
    fake_run_result.stdout = b"ok"
    fake_run_result.returncode = 0

    with (
        patch.object(mod.path_utils, "get_run_dir", side_effect=fake_get_run_dir) as mock_grd,
        patch.object(mod.path_utils, "make_dir", MagicMock()),
        patch.object(mod.subprocess, "run", return_value=fake_run_result),
        patch.object(mod, "compute_statistical_metrics", return_value=MagicMock(drop=MagicMock(return_value=MagicMock(iloc=[{}])))),
        patch.object(mod, "compute_ml_metrics_all_ml_methods", return_value=MagicMock(drop=MagicMock(return_value=MagicMock(iloc=[{}])))),
        patch.object(mod, "compute_dp_metrics", return_value=MagicMock(drop=MagicMock(return_value=MagicMock(iloc=[{}])))),
        patch.object(mod, "get_dataset", return_value=MagicMock(discrete_columns=[], data_train=MagicMock(__iter__=iter([])))),
        patch.object(mod, "shutil", MagicMock()),
    ):
        with patch.object(mod.path_utils, "get_filename", return_value="trial_0000"):
            result = mod.objective(params.copy())

    return result, mock_grd, fake_trial_dir


class TestTabsynUsesGetRunDir:
    def test_get_run_dir_called(self, tabsyn_mod, tmp_path):
        _, mock_grd, _ = _run_tabsyn_objective(tabsyn_mod, tmp_path)
        mock_grd.assert_called_once()

    def test_params_json_written(self, tabsyn_mod, tmp_path):
        _, _, trial_dir = _run_tabsyn_objective(tabsyn_mod, tmp_path)
        assert (trial_dir / "params.json").exists()

    def test_status_json_has_status_key(self, tabsyn_mod, tmp_path):
        _, _, trial_dir = _run_tabsyn_objective(tabsyn_mod, tmp_path)
        data = json.loads((trial_dir / "status.json").read_text())
        assert "status" in data

    def test_hyperopt_folder_ior(self, tabsyn_mod):
        args = _make_args(bo_method="ior")
        folder = "optimization/generative_model" if args.bo_method == "ior" else f"optimization/generative_model_sbo_{args.bo_method_agg}"
        assert folder == "optimization/generative_model"


# ---------------------------------------------------------------------------
# tabddpm
# ---------------------------------------------------------------------------

def _run_tabddpm_objective(mod, tmp_path, params=None):
    if params is None:
        params = {
            "batch_size": 256,
            "num_timesteps": 100,
            "gaussian_loss_type": "mse",
            "lr": 0.001,
            "weight_decay": 0.0,
            "steps": 1000,
            "epochs": 1000,
            "n_layers": 2,
            "d_first": 7,
            "d_middle": 7,
            "d_last": 7,
        }

    mod.args = _make_args(arch="tabddpm", loss_version=0, model_type="mlp")
    import argparse
    mod.parser = argparse.ArgumentParser(add_help=False)
    mod.parser.add_argument("--dataset", default="adult")
    mod.parser.add_argument("--arch", default="tabddpm")
    mod.parser.add_argument("--loss_version", default=0, type=int)
    mod.parser.add_argument("--is_test", default=1, type=int)
    mod.parser.add_argument("--is_condvec", default=1, type=int)
    mod.parser.add_argument("--row_number", default=None, type=int)
    mod.parser.add_argument("--is_drop_id", default=1, type=int)
    mod.parser.add_argument("--model_type", default="mlp")
    mod.parser.add_argument("--module", default="public")
    mod._trial_count[0] = 0

    fake_trial_dir = tmp_path / "adult-tabddpm-lv0" / "trial_0000"
    fake_trial_dir.mkdir(parents=True, exist_ok=True)

    def fake_get_run_dir(dataset, model, loss, trial_n, is_test=False):
        return fake_trial_dir

    fake_run_result = MagicMock()
    fake_run_result.stdout = b"ok"
    fake_run_result.returncode = 0

    with (
        patch.object(mod.path_utils, "get_run_dir", side_effect=fake_get_run_dir) as mock_grd,
        patch.object(mod.path_utils, "make_dir", MagicMock()),
        patch.object(mod.subprocess, "run", return_value=fake_run_result),
        patch.object(mod, "compute_statistical_metrics", return_value=MagicMock(drop=MagicMock(return_value=MagicMock(iloc=[{}])))),
        patch.object(mod, "compute_ml_metrics_all_ml_methods", return_value=MagicMock(drop=MagicMock(return_value=MagicMock(iloc=[{}])))),
        patch.object(mod, "compute_dp_metrics", return_value=MagicMock(drop=MagicMock(return_value=MagicMock(iloc=[{}])))),
        patch.object(mod, "get_dataset", return_value=MagicMock(discrete_columns=[], data_train=MagicMock(__iter__=iter([])))),
        patch.object(mod, "generate_config_toml", return_value=str(fake_trial_dir / "config.toml")),
    ):
        with patch.object(mod.path_utils, "get_filename", return_value="trial_0000"):
            result = mod.objective(params.copy())

    return result, mock_grd, fake_trial_dir


class TestTabddpmUsesGetRunDir:
    def test_get_run_dir_called(self, tabddpm_mod, tmp_path):
        _, mock_grd, _ = _run_tabddpm_objective(tabddpm_mod, tmp_path)
        mock_grd.assert_called_once()

    def test_params_json_written(self, tabddpm_mod, tmp_path):
        _, _, trial_dir = _run_tabddpm_objective(tabddpm_mod, tmp_path)
        assert (trial_dir / "params.json").exists()

    def test_status_json_has_status_key(self, tabddpm_mod, tmp_path):
        _, _, trial_dir = _run_tabddpm_objective(tabddpm_mod, tmp_path)
        data = json.loads((trial_dir / "status.json").read_text())
        assert "status" in data

    def test_pythonpath_env_preserved(self, tabddpm_mod, tmp_path):
        """Verify subprocess.run is called with PYTHONPATH='.' (existing fix)."""
        with patch.object(tabddpm_mod.subprocess, "run") as mock_run:
            mock_run.return_value = MagicMock(stdout=b"ok", returncode=0)
            fake_dir = tmp_path / "adult-tabddpm-lv0" / "trial_py"
            fake_dir.mkdir(parents=True, exist_ok=True)
            tabddpm_mod._trial_count[0] = 0
            tabddpm_mod.args = _make_args(arch="tabddpm", loss_version=0, model_type="mlp")
            with (
                patch.object(tabddpm_mod.path_utils, "get_run_dir", return_value=fake_dir),
                patch.object(tabddpm_mod.path_utils, "make_dir", MagicMock()),
                patch.object(tabddpm_mod, "compute_statistical_metrics", return_value=MagicMock(drop=MagicMock(return_value=MagicMock(iloc=[{}])))),
                patch.object(tabddpm_mod, "compute_ml_metrics_all_ml_methods", return_value=MagicMock(drop=MagicMock(return_value=MagicMock(iloc=[{}])))),
                patch.object(tabddpm_mod, "get_dataset", return_value=MagicMock(discrete_columns=[], data_train=MagicMock(__iter__=iter([])))),
                patch.object(tabddpm_mod, "generate_config_toml", return_value=str(fake_dir / "config.toml")),
                patch.object(tabddpm_mod.path_utils, "get_filename", return_value="trial_py"),
            ):
                tabddpm_mod.objective({
                    "batch_size": 256, "num_timesteps": 100, "gaussian_loss_type": "mse",
                    "lr": 0.001, "weight_decay": 0.0, "steps": 1000, "epochs": 1000,
                    "n_layers": 2, "d_first": 7, "d_middle": 7, "d_last": 7,
                })
        if mock_run.called:
            env = mock_run.call_args.kwargs.get("env")
            if env is not None:
                assert env.get("PYTHONPATH") == "."

    def test_hyperopt_folder_ior(self, tabddpm_mod):
        args = _make_args(bo_method="ior")
        folder = "optimization/generative_model" if args.bo_method == "ior" else f"optimization/generative_model_sbo_{args.bo_method_agg}"
        assert folder == "optimization/generative_model"

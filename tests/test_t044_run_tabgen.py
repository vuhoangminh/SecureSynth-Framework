"""Tests for t044: run_tabgen.py resolves dir_logs via get_run_dir()
and skips when the trial dir already contains synthetic output.
"""
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, create_autospec

import pytest


# ---------------------------------------------------------------------------
# module loader
# ---------------------------------------------------------------------------

def _load_run_tabgen():
    heavy = [
        "engine.ctgan_data_transformer",
        "models.ctgan",
        "models.tvae",
        "models.copulagan",
        "models.dpcgans",
        "models.CTAB",
        "models.CTAB.ctabgan",
        "sdv",
        "sdv.metadata",
    ]
    originals = {}
    for name in heavy:
        if name not in sys.modules:
            sys.modules[name] = MagicMock()
            originals[name] = None
        else:
            originals[name] = sys.modules[name]
            sys.modules[name] = MagicMock()

    # engine.datasets uses star-import; give it a real get_dataset so the module namespace gets it
    datasets_mock = types.ModuleType("engine.datasets")
    datasets_mock.get_dataset = MagicMock()
    originals["engine.datasets"] = sys.modules.get("engine.datasets")
    sys.modules["engine.datasets"] = datasets_mock

    spec = importlib.util.spec_from_file_location("_run_tabgen", "scripts/optimize/run_tabgen.py")
    mod = importlib.util.module_from_spec(spec)
    # parse_args() is called at module level; prevent pytest argv from leaking in
    orig_argv = sys.argv[:]
    sys.argv = ["run_tabgen.py"]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = orig_argv

    for name, original in originals.items():
        if original is None:
            del sys.modules[name]
        else:
            sys.modules[name] = original

    return mod


@pytest.fixture(scope="module")
def rtg():
    return _load_run_tabgen()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_rtg_args(rtg, trial_n=0, **kwargs):
    import argparse
    ns = argparse.Namespace(
        dataset="adult",
        arch="ctgan",
        loss_version=2,
        is_test=1,
        trial_n=trial_n,
        row_number=None,
        row_number_full=None,
        is_drop_id=1,
        is_loss_corr=1.0,
        is_loss_dwp=1.0,
        n_moment_loss_dwp=4,
        batch_size=500,
        epochs=20,
        embedding_dim=64,
        generator_dim=64,
        discriminator_dim=64,
        generator_lr=2e-4,
        generator_decay=1e-6,
        discriminator_lr=2e-4,
        discriminator_decay=1e-6,
        discriminator_steps=1,
        dp_sigma=1.0,
        dp_weight_clip=0.01,
        private=0,
        is_condvec=1,
        is_only_sample=0,
        checkpoint_freq=50,
        resume=1,
        n_run=2,
        print_freq=10,
        dir_logs="database/gan/",
    )
    for k, v in kwargs.items():
        setattr(ns, k, v)
    return ns


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

class TestRunTabgenUsesGetRunDir:
    def test_get_run_dir_called_in_main(self, rtg, tmp_path):
        """main() must call get_run_dir to build dir_logs."""
        import pandas as pd

        fake_dir = tmp_path / "adult-ctgan-lv2" / "trial_0000"
        fake_dir.mkdir(parents=True, exist_ok=True)

        rtg.args = _make_rtg_args(rtg)

        fake_dataset = MagicMock()
        fake_dataset.data_train = pd.DataFrame({"a": range(10), "b": range(10)})
        fake_dataset.discrete_columns = []
        fake_dataset.target = "b"

        def fake_get_run_dir(dataset, model, loss, trial_n, is_test=False):
            return fake_dir

        with (
            patch.object(rtg.path_utils, "get_run_dir", side_effect=fake_get_run_dir) as mock_grd,
            patch.object(rtg.path_utils, "make_dir", MagicMock()),
            patch("builtins.open", MagicMock()),
            patch.object(rtg, "get_dataset", return_value=fake_dataset),
            patch.object(rtg, "data_utils", MagicMock()),
            patch.object(rtg, "print_utils", MagicMock()),
            patch.object(rtg, "logger", MagicMock()),
            patch.object(rtg, "model_utils", MagicMock()),
            patch.object(rtg, "sys", MagicMock(exit=lambda x: (_ for _ in ()).throw(SystemExit(x)))),
        ):
            try:
                rtg.main()
            except (SystemExit, Exception):
                pass  # we only care that get_run_dir was called

        mock_grd.assert_called_once()
        call_args = mock_grd.call_args[0]
        assert call_args[0] == "adult"   # dataset
        assert call_args[1] == "ctgan"   # arch
        assert call_args[2] == "lv2"     # loss
        assert call_args[3] == 0         # trial_n

    def test_dir_logs_set_from_get_run_dir(self, rtg, tmp_path):
        """args.dir_logs must match the path returned by get_run_dir."""
        import pandas as pd

        fake_dir = tmp_path / "adult-ctgan-lv2" / "trial_0007"
        fake_dir.mkdir(parents=True, exist_ok=True)

        rtg.args = _make_rtg_args(rtg, trial_n=7)

        fake_dataset = MagicMock()
        fake_dataset.data_train = pd.DataFrame({"a": range(10), "b": range(10)})
        fake_dataset.discrete_columns = []
        fake_dataset.target = "b"

        def fake_get_run_dir(dataset, model, loss, trial_n, is_test=False):
            return fake_dir

        with (
            patch.object(rtg.path_utils, "get_run_dir", side_effect=fake_get_run_dir),
            patch.object(rtg.path_utils, "make_dir", MagicMock()),
            patch("builtins.open", MagicMock()),
            patch.object(rtg, "get_dataset", return_value=fake_dataset),
            patch.object(rtg, "data_utils", MagicMock()),
            patch.object(rtg, "print_utils", MagicMock()),
            patch.object(rtg, "logger", MagicMock()),
            patch.object(rtg, "model_utils", MagicMock()),
            patch.object(rtg, "sys", MagicMock(exit=lambda x: (_ for _ in ()).throw(SystemExit(x)))),
        ):
            try:
                rtg.main()
            except (SystemExit, Exception):
                pass

        assert rtg.args.dir_logs == str(fake_dir)


class TestRunTabgenSkipIfExists:
    def test_exits_early_when_synthetic_full_exists(self, rtg, tmp_path):
        """main() must sys.exit(0) when synthetic_full.csv already exists in the trial dir."""
        import pandas as pd

        fake_dir = tmp_path / "adult-ctgan-lv2" / "trial_0000"
        fake_dir.mkdir(parents=True, exist_ok=True)
        (fake_dir / "synthetic_full.csv").write_text("col\n1\n")

        rtg.args = _make_rtg_args(rtg)

        fake_dataset = MagicMock()
        fake_dataset.data_train = pd.DataFrame({"a": range(10), "b": range(10)})
        fake_dataset.discrete_columns = []
        fake_dataset.target = "b"

        exited = []

        def fake_exit(code):
            exited.append(code)
            raise SystemExit(code)

        with (
            patch.object(rtg.path_utils, "get_run_dir", return_value=fake_dir),
            patch.object(rtg.path_utils, "make_dir", MagicMock()),
            patch.object(rtg, "get_dataset", return_value=fake_dataset),
            patch.object(rtg, "data_utils", MagicMock()),
            patch.object(rtg, "print_utils", MagicMock()),
            patch.object(rtg, "logger", MagicMock()),
            patch.object(rtg, "sys") as mock_sys,
        ):
            mock_sys.exit.side_effect = fake_exit
            with pytest.raises(SystemExit):
                rtg.main()

        assert exited == [0]

    def test_proceeds_when_trial_dir_absent(self, rtg, tmp_path):
        """main() must not exit early when the trial dir has no synthetic output."""
        import pandas as pd

        fake_dir = tmp_path / "adult-ctgan-lv2" / "trial_0001"
        fake_dir.mkdir(parents=True, exist_ok=True)
        # no synthetic_full.csv or df_score_data_sufficient.csv

        rtg.args = _make_rtg_args(rtg, trial_n=1)

        fake_dataset = MagicMock()
        fake_dataset.data_train = pd.DataFrame({"a": range(10), "b": range(10)})
        fake_dataset.discrete_columns = []
        fake_dataset.target = "b"

        exited_with_zero = []

        def fake_exit(code):
            if code == 0:
                exited_with_zero.append(code)
            raise SystemExit(code)

        fake_model = MagicMock()
        fake_model.fit = MagicMock()

        with (
            patch.object(rtg.path_utils, "get_run_dir", return_value=fake_dir),
            patch.object(rtg.path_utils, "make_dir", MagicMock()),
            patch.object(rtg, "get_dataset", return_value=fake_dataset),
            patch.object(rtg, "data_utils", MagicMock()),
            patch.object(rtg, "print_utils", MagicMock()),
            patch.object(rtg, "logger", MagicMock()),
            patch.object(rtg, "model_utils", MagicMock()),
            patch.object(rtg, "CTGAN", return_value=fake_model),
            patch.object(rtg, "sys") as mock_sys,
        ):
            mock_sys.exit.side_effect = fake_exit
            try:
                rtg.main()
            except (SystemExit, Exception):
                pass

        assert exited_with_zero == [], "should not have exited early with code 0"

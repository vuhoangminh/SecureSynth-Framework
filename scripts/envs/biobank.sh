conda create --solver=libmamba -y -n biobank312 -c rapidsai -c conda-forge -c nvidia rapids=25.04 python=3.10 'cuda-version>=12.0,<=12.8' jupyterlab dash

# conda activate doesn't work in non-interactive bash subshells; source the hook first
eval "$(conda shell.bash hook)"
conda activate biobank312
export PIP_NO_INPUT=1

# Install all the dependencies

# Pin numpy before any package that constrains it (<2)
pip install "numpy==1.26.4"

# Install torch before ctgan — ctgan unversioned pulls torch 2.12+cu130 which needs
# driver >=565; CUDA 12.2 (driver 535) supports cu121 at most (max torch 2.5.1).
# opacus 1.5.4 requires torch>=2.2.0; monai 1.5.2 requires torch>=2.4.1 — both satisfied.
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

pip install ctgan

# imgui and pyspng require system package libspng-dev and are only needed for
# StyleGAN-style visualization — skip on headless/server setups.
# To enable: sudo apt-get install -y libspng-dev, then uncomment the line below.
# pip install imgui==1.3.0 glfw==2.2.0 pyopengl==3.1.5 imageio imageio-ffmpeg==0.4.4 pyspng==0.1.0 click
pip install imageio "imageio-ffmpeg==0.4.4" click requests psutil

pip install scipy monai

# albumentations -U upgrades numpy to 2.x; re-pin immediately after
pip install albumentations
pip install "numpy==1.26.4"

pip install nilearn
pip install SimpleITK
pip install comet-ml

# opacus 1.6.0 requires torch>=2.6.0 which exceeds cu121 max (2.5.1); use 1.5.4
pip install opacus==1.5.4

pip install matplotlib
pip install seaborn
# conda installs xgboost + libxgboost (via RAPIDS) — remove both first to avoid
# .so version mismatch with the pip-pinned version
conda remove -y xgboost libxgboost --force
pip install "xgboost==1.7.6"
pip install autoflake
pip install termcolor
pip install hyperopt
pip install "sdv==1.8.0"
pip install "rdt==1.21.0" "numpy==1.26.4"
pip install wandb
pip install scikit_posthocs
pip install imbalanced-learn
pip install tabulate
pip install researchpy
pip install icecream
pip install category_encoders
pip install tomli_w
# libzero and rtdl both hard-pin torch<2; install without deps to keep torch 2.5.1
pip install "libzero==0.0.8" --no-deps
pip install "rtdl==0.0.9" --no-deps
pip install "catboost==1.0.3"
pip install "anonymeter==1.0.0"

pip install "pandas==1.5.3"
# RAPIDS conda installs xarray 2025.x which requires pandas>=2.1; override with a
# version compatible with pandas 1.5.3 so that sdv/plotly imports don't crash.
pip install "xarray<2024"
pip install xlrd

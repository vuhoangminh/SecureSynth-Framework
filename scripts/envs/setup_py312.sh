wget "https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh"
bash -b -p Anaconda3-2024.06-1-Linux-x86_64.sh

conda create --solver=libmamba -y -n biobank312 -c rapidsai -c conda-forge -c nvidia rapids=25.04 python=3.10 'cuda-version>=12.0,<=12.8' jupyterlab dash
conda activate biobank312

# Install all the dependencies
conda install cuda-cudart cuda-version=12
pip install ctgan
pip install imgui==1.3.0 glfw==2.2.0 pyopengl==3.1.5 imageio imageio-ffmpeg==0.4.4 pyspng==0.1.0 click requests psutil
pip install scipy monai
pip install -U albumentations
pip install nilearn
pip install SimpleITK
pip install comet-ml
pip install opacus
pip install matplotlib
pip install seaborn
pip install xgboost==1.7.6
pip install autoflake
pip install termcolor
pip install hyperopt
pip install sdv==1.8.0
pip install wandb
pip uninstall -y torch
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install scikit_posthocs
pip install imbalanced-learn
pip install tabulate
pip install researchpy
pip install icecream
pip install category_encoders
pip install tomli_w
pip install libzero==0.0.8
pip install rtdl==0.0.9
pip install catboost==1.0.3
pip install anonymeter==1.0.0


# Uninstall existing version of pandas and install version 1.5.3
pip install -y pandas=1.5.3

pip install xlrd




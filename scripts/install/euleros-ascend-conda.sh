#!/bin/bash
set -ex

MINDSPORE_VERSION=${MINDSPORE_VERSION:-1.5.0}
PYTHON_VERSION=${PYTHON_VERSION:-3.7.5}
MINDSPORE_VERSION=${MINDSPORE_VERSION:-1.5.0}

cd /tmp
curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.10.3-Linux-aarch64.sh
bash Miniconda3-py37_4.10.3-Linux-aarch64.sh -b

# add conda to PATH
echo -e 'export PATH=~/miniconda3/bin/:$PATH' >> ~/.bash_profile
echo -e '. ~/miniconda3/etc/profile.d/conda.sh' >> ~/.bash_profile
source ~/.bash_profile
conda init bash
# setting up conda mirror
cat >~/.condarc  <<END
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
END

# initialize conda env and install mindspore-ascend
conda create -n ms_${PYTHON_VERSION} python=${PYTHON_VERSION} -y
conda activate ms_${PYTHON_VERSION}
conda install mindspore-ascend=${MINDSPORE_VERSION} -c mindspore -c conda-forge

# verify installation
python -c "import mindspore;mindspore.run_check()"
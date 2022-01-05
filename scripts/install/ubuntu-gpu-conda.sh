#!/bin/bash
set -ex

MINDSPORE_VERSION=${MINDSPORE_VERSION:-1.5.0}
PYTHON_VERSION=${PYTHON_VERSION:-3.7.5}
MINDSPORE_VERSION=${MINDSPORE_VERSION:-1.5.0}
CUDA_VERSION=${CUDA_VERSION:-11.1.1-1}
LIB_CUDA_VERSION=${LIB_CUDA_VERSION:-8.0.5.39-1+cuda11.1}
DISTRIBUTED=${DISTRIBUTED:-false}
CUDATOOLKIT_VERSION=${CUDATOOLKIT_VERSION:-11.1}
CUDNN_VERSION=${CUDNN_VERSION:-8.0.5}

cd /tmp
curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh
bash Miniconda3-py37_4.10.3-Linux-x86_64.sh

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

#initialize conda env and install mindspore-cpu

conda create -n ms_${PYTHON_VERSION} python=${PYTHON_VERSION} -y
conda activate ms_${PYTHON_VERSION}

# install gmp 6.1.2, downloading gmp is slow
# echo "install gmp start"
# sudo apt-get install m4 -y 
# cd /tmp
# curl -O https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz
# xz -d gmp-6.1.2.tar.xz
# tar xvzf gmp-6.1.2.tar && cd gmp-6.1.2
# ./configure --prefix=/usr/local/gmp-6.1.2
# make
# sudo make install
# echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/gmp-6.1.2/lib' >> ~/.bash_profile

# install mindspore-gpu with conda
conda install mindspore-gpu=${MINDSPORE_VERSION} cudatoolkit=${CUDATOOLKIT_VERSION} -c mindspore -c conda-forge -y

# check if it is the right mindspore version
python -c "import mindspore;mindspore.run_check()"

# check if it can be run with GPU

cat > example.py <<END
import numpy as np
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.context as context

context.set_context(device_target="GPU")
x = Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(ops.add(x, y))
END

python example.py
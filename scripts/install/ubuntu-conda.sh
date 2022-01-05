#!/bin/bash
set -ex

MINDSPORE_VERSION=${MINDSPORE_VERSION:-1.5.0}
PYTHON_VERSION=${PYTHON_VERSION:-3.7.5}

#use huaweicloud mirror in China
sudo sed -i "s@http://.*archive.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list
sudo sed -i "s@http://.*security.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list
sudo apt-get update

# install python 3.7 and make it default
sudo apt-get install gcc-7 libgmp-dev curl python3.7  -y
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.7 100

cd /tmp
curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh
bash Miniconda3-py37_4.10.3-Linux-x86_64.sh -b 

# add conda to PATH
 echo -e 'export PATH=~/miniconda3/bin/:$PATH' >> ~/.bash_profile
 echo -e '. ~/miniconda3/etc/profile.d/conda.sh' >> ~/.bash_profile
 source ~/.bash_profile
conda init bash
# setting up conda mirror with qinghua source
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
conda install mindspore-cpu=${MINDSPORE_VERSION} -c mindspore -c conda-forge -y

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
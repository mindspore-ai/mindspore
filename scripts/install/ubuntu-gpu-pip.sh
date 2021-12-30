#!/bin/bash
set -ex

PYTHON_VERSION=${PYTHON_VERSION:-3.7.5}
MINDSPORE_VERSION=${MINDSPORE_VERSION:-1.5.0}
CUDA_VERSION=${CUDA_VERSION:-8.0.5.39-1+cuda11.1}
DISTRIBUTED=${DISTRIBUTED:-false}
CUDA_VERSION=${CUDA_VERSION:-11.1.1-1}
CUDA_INSTALL_PATH=${CUDA_INSTALL_PATH:-cuda-11.1}
LIBNCCL2_VERSION=${LIBNCCL2_VERSION:-2.7.8-1+cuda11.1}
ARCH=$(uname -m)

declare -A version_map=()
version_map["3.7.5"]="${MINDSPORE_VERSION}-cp37-cp37m"
version_map["3.9.0"]="${MINDSPORE_VERSION}-cp39-cp39m"

#use huaweicloud mirror in China
sudo sed -i "s@http://.*archive.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list
sudo sed -i "s@http://.*security.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list
sudo apt-get update

# install python 3.7 and make it default
sudo apt-get install gcc-7 libgmp-dev curl python3.7  openssl ubuntu-drivers-common openssl software-properties-common -y
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.7 100

cd /tmp
curl -O https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py

# add pip mirror
mkdir -p ~/.pip
cat > ~/.pip/pip.conf <<END
[global]
index-url = https://repo.huaweicloud.com/repository/pypi/simple
trusted-host = repo.huaweicloud.com
timeout = 120
END

# install nvidia driver if not presented
# root@ecs-gpu-testing:~# ubuntu-drivers devices
# == /sys/devices/pci0000:20/0000:20:00.0/0000:21:01.0 ==
# modalias : pci:v000010DEd00001EB8sv000010DEsd000012A2bc03sc02i00
# vendor   : NVIDIA Corporation
# driver   : nvidia-driver-418 - third-party non-free
# driver   : nvidia-driver-450 - third-party non-free
# driver   : nvidia-driver-460 - third-party non-free
# driver   : nvidia-driver-450-server - distro non-free
# driver   : nvidia-driver-460-server - distro non-free
# driver   : nvidia-driver-440 - third-party non-free
# driver   : nvidia-driver-418-server - distro non-free
# driver   : nvidia-driver-465 - third-party non-free
# driver   : nvidia-driver-470 - third-party non-free recommended #pick the latest one
# driver   : nvidia-driver-410 - third-party non-free
# driver   : nvidia-driver-470-server - distro non-free
# driver   : nvidia-driver-455 - third-party non-free
# driver   : xserver-xorg-video-nouveau - distro free builtin
# sudo apt-get install nvidia-driver-470 -y
# nvidia-smi # run this to check the driver is working
#root@ecs-testing:~# nvidia-smi
#Thu Dec 30 21:06:13 2021
#+-----------------------------------------------------------------------------+
#| NVIDIA-SMI 460.73.01    Driver Version: 460.73.01    CUDA Version: 11.2     |
#|-------------------------------+----------------------+----------------------+
#| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
#| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
#|                               |                      |               MIG M. |
#|===============================+======================+======================|
#|   0  Tesla T4            Off  | 00000000:21:01.0 Off |                    0 |
#| N/A   61C    P0    29W /  70W |      0MiB / 15109MiB |      0%      Default |
#|                               |                      |                  N/A |
#+-------------------------------+----------------------+----------------------+
#
#+-----------------------------------------------------------------------------+
#| Processes:                                                                  |
#|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
#|        ID   ID                                                   Usage      |
#|=============================================================================|
#|  No running processes found                                                 |
#+-----------------------------------------------------------------------------+

# install cuda/cudnn/nccl2 with apt-get
# another option is to use linux.run https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda_11.1.0_455.23.05_linux.run
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo add-apt-repository "deb https://developer.download.nvidia.cn/compute/machine-learning/repos/ubuntu1804/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda=${CUDA_VERSION}

# add cuda to path 
cat >> ~/.bash_profile <<END
export PATH=/usr/local/cuda/bin:\$PATH
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda/lib64
END
source ~/.bash_profile
echo "cuda install success."

sudo apt-get install -y libcudnn8=${CUDA_VERSION} libcudnn8-dev=${CUDA_VERSION} libnccl2=${LIBNCCL2_VERSION} libnccl-dev=${LIBNCCL2_VERSION}

# optional (tensort for serving, openmpi for distributed training)
# uncomment this to compile openmpi
#   cd /tmp
#   curl -O https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.3.tar.gz
#   tar xvzf openmpi-4.0.3.tar.gz
#   cd openmpi-4.0.3
#   ./configure --prefix=/usr/local/openmpi
#   make 
#   sudo make install 
#   echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/openmpi/lib' >> ~/.bash_profile
#   echo 'export PATH=$PATH:/usr/local/openmpi/bin' >> ~/.bash_profile
#   source ~/.bash_profile
#  reference this to install tensorrt https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#downloading

echo "install mindspore gpu ${MINDSPORE_VERSION}"
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MINDSPORE_VERSION}/MindSpore/gpu/${ARCH}/${CUDA_INSTALL_PATH}/mindspore_gpu-${version_map["$PYTHON_VERSION"]}-linux_${ARCH}.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple


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
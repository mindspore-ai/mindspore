#!/bin/bash
set -ex

PYTHON_VERSION=${PYTHON_VERSION:-3.7.5}
MINDSPORE_VERSION=${MINDSPORE_VERSION:-1.5.0}
CUDA_VERSION=${CUDA_VERSION:-8.0.5.39-1+cuda11.1}
DISTRIBUTED=${DISTRIBUTED:-false}
CUDA_INSTALL_PATH=${CUDA_INSTALL_PATH:-cuda-11}
ARCH=$(uname -m)

if [[ "${PYTHON_VERSION}" == "3.7.5" ]]; then
VERSION="${MINDSPORE_VERSION}-cp37-cp37m"
else
VERSION="${MINDSPORE_VERSION}-cp39-cp39m"
fi

#use huaweicloud mirror in China
sudo sed -i "s@http://.*archive.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list
sudo sed -i "s@http://.*security.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list
sudo apt-get update

# install python 3.7 and make it default
sudo apt-get install gcc-7 libgmp-dev curl python3.7  openssl ubuntu-drivers-common openssl -y
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.7 100

cd /tmp
curl -O https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py

# add pip mirror

mkdir ~/.pip

cat > ~/.pip/pip.conf <<END
[global]
index-url = https://repo.huaweicloud.com/repository/pypi/simple
trusted-host = repo.huaweicloud.com
timeout = 120
END

# install gmp 6.1.2, downloading gmp is slow

sudo apt-get install m4 -y 
cd /tmp
curl -O https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz
xz -d gmp-6.1.2.tar.xz
tar xvzf gmp-6.1.2.tar && cd gmp-6.1.2
./configure --prefix=/usr/local/gmp-6.1.2
make
sudo make install
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/gmp-6.1.2/lib' >> ~/.bash_profile

# install cuda with linux.run
# another option is to use https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda_11.1.0_455.23.05_linux.run

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget -c https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda-repo-ubuntu1804-11-1-local_11.1.1-455.32.00-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-11-1-local_11.1.1-455.32.00-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu1804-11-1-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda

# add cuda to path 
cat >> ~/.bash_profile <<END
export PATH=/usr/local/cuda/bin:\$PATH
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda/lib64
END
source ~/.bash_profile

echo "cuda install success."

# install cudnn
cd /tmp
wget https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu1804/x86_64/libcudnn8_${CUDA_VERSION}_amd64.deb
wget https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu1804/x86_64/libcudnn8-dev_${CUDA_VERSION}_amd64.deb
sudo dpkg -i libcudnn8_${CUDA_VERSION}_amd64.deb libcudnn8-dev_${CUDA_VERSION}_amd64.deb

# Install CuDNN 8 and NCCL 2
wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo dpkg -i nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt update
sudo apt install -y libcudnn8=${CUDA_VERSION} libcudnn8-dev=${CUDA_VERSION} libnccl2=2.7.8-1+cuda11.1 libnccl2-dev=2.7.8-1+cuda11.1 

# optional (tensort for serving, openmpi for distributed training)
if [[ "$DISTRIBUTED" == "false" ]]; then
    cd /tmp
    curl -O https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.3.tar.gz
    tar xvzf openmpi-4.0.3.tar.gz
    cd openmpi-4.0.3
    ./configure --prefix=/usr/local/openmpi
    make 
    sudo make install 
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/openmpi/lib' >> ~/.bash_profile
    echo 'export PATH=$PATH:/usr/local/openmpi/bin' >> ~/.bash_profile
    source ~/.bash_profile
fi

# install mindspore gpu

pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MINDSPORE_VERSION}/MindSpore/gpu/${CUDA_INSTALL_PATH}/mindspore-${VERSION}-linux_${ARCH}.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
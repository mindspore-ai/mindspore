#!/bin/bash
set -ex



PYTHON_VERSION=${PYTHON_VERSION:-3.7.5}
MINDSPORE_VERSION=${MINDSPORE_VERSION:-1.5.0}

declare -A map


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
sudo apt-get install gcc-7 libgmp-dev curl python3.7  openssl -y
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.7 100

cd /tmp
curl -O https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py


# install cuda 

# cd /tmp
# wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda_11.1.0_455.23.05_linux.run


wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget -c https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda-repo-ubuntu1804-11-1-local_11.1.1-455.32.00-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-11-1-local_11.1.1-455.32.00-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu1804-11-1-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda

# cuda to add path 

cat >> ~/.bash_profile <<END
export PATH=/usr/local/cuda/bin:\$PATH
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda/lib64
END

source ~/.bash_profile

[[ nvcc -V ]] && echo "cuda install success."

# install cudnn
cd /tmp
wget https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu1804/x86_64/libcudnn8_8.0.5.39-1+cuda11.1_amd64.deb
wget https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu1804/x86_64/libcudnn8-dev_8.0.5.39-1+cuda11.1_amd64.deb

sudo dpkg -i libcudnn8_8.0.5.39-1+cuda11.1_amd64.deb libcudnn8-dev_8.0.5.39-1+cuda11.1_amd64.deb


# reference 
# - https://gist.github.com/bogdan-kulynych/f64eb148eeef9696c70d485a76e42c3a
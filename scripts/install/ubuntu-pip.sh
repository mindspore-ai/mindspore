#!/bin/bash
set -ex

# sudo cp -a /etc/apt/sources.list /etc/apt/sources.list.bak  单独执行
PYTHON_VERSION=${PYTHON_VERSION:-3.7.5}
MINDSPORE_VERSION=${MINDSPORE_VERSION:-1.5.0}
ARCH=`uname -m`

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
sudo apt-get install gcc-7 libgmp-dev curl python3.7  -y
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.7 100

cd /tmp
curl -O https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py

pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MINDSPORE_VERSION}/MindSpore/cpu/${ARCH}/mindspore-${VERSION}-linux_${ARCH}.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
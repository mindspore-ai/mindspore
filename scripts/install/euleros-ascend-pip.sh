#!/bin/bash
set -ex

# sudo cp -a /etc/apt/sources.list /etc/apt/sources.list.bak  单独执行
PYTHON_VERSION=${PYTHON_VERSION:-3.7.5}
MINDSPORE_VERSION=${MINDSPORE_VERSION:-1.5.0}
ARCH=`uname -m`

if [[ "${PYTHON_VERSION}" == "3.7.5" ]]; then
    VERSION="${MINDSPORE_VERSION}-cp37-cp37m"
else
    VERSION="${MINDSPORE_VERSION}-cp39-cp39"
fi

#use huaweicloud mirror in China
repo_path=/etc/yum.repos.d/euleros.repo
cat > ${repo_path} << END
[base]
name=EulerOS-2.0SP8 base
baseurl=http://repo.huaweicloud.com/euler/2.8/os/${ARCH}
enabled=1
gpgcheck=1
gpgkey=http://repo.huaweicloud.com/euler/2.8/os/RPM-GPG-KEY-EulerOS
END
cat ${repo_path}

yum clean all
yum makecache

yum install gmp-devel
yum install 

# install python 3.7 
cd /tmp
wget https://github.com/python/cpython/archive/v3.7.5.tar.gz
tar -xvf v3.7.5.tar.gz
cd /tmp/cpython-3.7.5
mkdir -p ${PYTHON_ROOT_PATH}
./configure --prefix=${PYTHON_ROOT_PATH} --enable-shared
make -j4
make install -j4
rm -f /usr/local/bin/python
rm -f /usr/local/bin/pip
rm -f /usr/local/lib/libpython3.7m.so.1.0
ln -s ${PYTHON_ROOT_PATH}/bin/python3.7 /usr/local/bin/python
ln -s ${PYTHON_ROOT_PATH}/bin/pip3.7 /usr/local/bin/pip
ln -s ${PYTHON_ROOT_PATH}/lib/libpython3.7m.so.1.0 /usr/local/lib/libpython3.7m.so.1.0
ldconfig
rm -rf /tmp/cpython-3.7.5
rm -f /tmp/v3.7.5.tar.gz

pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MINDSPORE_VERSION}/MindSpore/ascend/${ARCH}/mindspore-${VERSION}-linux_${ARCH}.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
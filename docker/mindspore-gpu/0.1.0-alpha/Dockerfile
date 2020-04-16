FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

MAINTAINER leonwanghui <leon.wanghui@huawei.com>

# Set env
ENV PYTHON_ROOT_PATH /usr/local/python-3.7.5
ENV OMPI_ROOT_PATH /usr/local/openmpi-3.1.5
ENV PATH ${OMPI_ROOT_PATH}/bin:/usr/local/bin:$PATH
ENV LD_LIBRARY_PATH ${OMPI_ROOT_PATH}/lib:$LD_LIBRARY_PATH

# Install base tools
RUN apt update \
    && DEBIAN_FRONTEND=noninteractive apt install -y \
    vim \
    wget \
    curl \
    xz-utils \
    net-tools \
    openssh-client \
    git \
    ntpdate \
    tzdata \
    tcl \
    sudo \
    bash-completion

# Install compile tools
RUN DEBIAN_FRONTEND=noninteractive apt install -y \
    gcc \
    g++ \
    zlibc \
    make \
    libgmp-dev \
    patch \
    autoconf \
    libtool \
    automake \
    flex \
    libnccl2=2.4.8-1+cuda10.1 \
    libnccl-dev=2.4.8-1+cuda10.1

# Set bash
RUN echo "dash dash/sh boolean false" | debconf-set-selections
RUN DEBIAN_FRONTEND=noninteractive dpkg-reconfigure dash

# Install python (v3.7.5)
RUN apt install -y libffi-dev libssl-dev zlib1g-dev libbz2-dev libncurses5-dev \
    libgdbm-dev libgdbm-compat-dev liblzma-dev libreadline-dev libsqlite3-dev \
    && cd /tmp \
    && wget https://github.com/python/cpython/archive/v3.7.5.tar.gz \
    && tar -xvf v3.7.5.tar.gz \
    && cd /tmp/cpython-3.7.5 \
    && mkdir -p ${PYTHON_ROOT_PATH} \
    && ./configure --prefix=${PYTHON_ROOT_PATH} \
    && make -j4 \
    && make install -j4 \
    && rm -f /usr/local/bin/python \
    && rm -f /usr/local/bin/pip \
    && ln -s ${PYTHON_ROOT_PATH}/bin/python3.7 /usr/local/bin/python \
    && ln -s ${PYTHON_ROOT_PATH}/bin/pip3.7 /usr/local/bin/pip \
    && rm -rf /tmp/cpython-3.7.5 \
    && rm -f /tmp/v3.7.5.tar.gz

# Set pip source
RUN mkdir -pv /root/.pip \
    && echo "[global]" > /root/.pip/pip.conf \
    && echo "trusted-host=mirrors.aliyun.com" >> /root/.pip/pip.conf \
    && echo "index-url=http://mirrors.aliyun.com/pypi/simple/" >> /root/.pip/pip.conf

# Install openmpi (v3.1.5)
RUN cd /tmp \
    && wget https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.5.tar.gz \
    && tar -xvf openmpi-3.1.5.tar.gz \
    && cd /tmp/openmpi-3.1.5 \
    && mkdir -p ${OMPI_ROOT_PATH} \
    && ./configure --prefix=${OMPI_ROOT_PATH} \
    && make -j4 \
    && make install -j4 \
    && rm -rf /tmp/openmpi-3.1.5 \
    && rm -f /tmp/openmpi-3.1.5.tar.gz

# Install MindSpore cuda-10.1 whl package
RUN pip install --no-cache-dir https://ms-release.obs.cn-north-4.myhuaweicloud.com/0.1.0-alpha/MindSpore/gpu/cuda-10.1/mindspore-0.1.0-cp37-cp37m-linux_x86_64.whl

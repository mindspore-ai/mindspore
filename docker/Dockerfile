# ===================== 构建参数 =====================
ARG CANN_VERSION
ARG CHIP_ARCH
ARG OS_SYSTEM
ARG PY_VERSION

# ===================== CANN 官方基础镜像 =====================

FROM quay.io/ascend/cann:${CANN_VERSION}-${CHIP_ARCH}-${OS_SYSTEM}-${PY_VERSION} AS base

# ===================== 安装阶段 =====================

## set pip source
ARG PIP_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple
RUN pip3 config set global.index-url ${PIP_INDEX_URL}

##  install pip package for mindspore
RUN pip3 install wheel \
    && pip3 install -U setuptools \
    && pip3 install --no-cache-dir pyyaml \
    && pip3 install  --no-cache-dir "numpy>=1.19.3,<=1.26.4" \
    && pip3 install  --no-cache-dir sympy \
    && pip3 install  --no-cache-dir decorator \
    && pip3 install  --no-cache-dir protobuf \
    && pip3 install  --no-cache-dir attrs

# install mindspore pkg
ARG MINDSPORE_VERSION
RUN pip3 install --no-cache-dir mindspore==${MINDSPORE_VERSION} -i https://repo.mindspore.cn/pypi/simple --trusted-host repo.mindspore.cn --extra-index-url ${PIP_INDEX_URL}

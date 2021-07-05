# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
ARG FROM_IMAGE_NAME
FROM ${FROM_IMAGE_NAME}

ARG SDK_PKG

RUN ln -s  /usr/local/python3.7.5/bin/python3.7 /usr/bin/python

RUN apt-get update && \
    apt-get install libglib2.0-dev -y || \
    rm -rf /var/lib/dpkg/info && \
    mkdir /var/lib/dpkg/info && \
    apt-get install libglib2.0-dev -y && \
    pip install pytest-runner==5.3.0

# pip install sdk_run
COPY $SDK_PKG .
RUN ls -hrlt
RUN chmod +x ${SDK_PKG} && \
    ./${SDK_PKG}  --install-path=/home/run --install && \
     bash -c "source ~/.bashrc"
#!/bin/bash
# Copyright 2019 Huawei Technologies Co., Ltd
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
export LOCAL_HIAI=/usr/local/Ascend
export TBE_IMPL_PATH=${LOCAL_HIAI}/latest/opp/built-in/op_impl/ai_core/tbe/impl/
export LD_LIBRARY_PATH=${LOCAL_HIAI}/latest/lib64/:${LD_LIBRARY_PATH}
export PATH=${LOCAL_HIAI}/latest/compiler/ccec_compiler/bin/:${PATH}
export PYTHONPATH=${LOCAL_HIAI}/latest/opp/built-in/op_impl/ai_core/tbe/:${PYTHONPATH}

set -e
BASEPATH=$(cd "$(dirname $0)"; pwd)
rm -rf "${BASEPATH}/mem_reuse_check/"
mkdir "${BASEPATH}/mem_reuse_check/"
# 1. run normal && check file exist
python "${BASEPATH}"/resnet_cifar_normal.py
if [ $? -ne 0 ]; then
    echo "[ERROR] resnet_cifar_normal run failed"
    exit 1
fi
# 2.  copy normal to current dir
mv "./normal_mem.ir" "${BASEPATH}/mem_reuse_check/"
# 3. run memreuse && check file exist
python "${BASEPATH}"/resnet_cifar_memreuse.py
if [ $? -ne 0 ]; then
    echo "[ERROR] resnet_cifar_memreuse run failed"
    exit 1
fi
# 4. copy memreuse ir to current dir
mv "./memreuse.ir" "${BASEPATH}/mem_reuse_check/"
# 5. check file whether same && return true
python "${BASEPATH}"/check_file.py
if [ $? -ne 0 ]; then
    echo "[ERROR] check_file run failed"
    exit 1
fi
rm -rf "${BASEPATH}/mem_reuse_check"

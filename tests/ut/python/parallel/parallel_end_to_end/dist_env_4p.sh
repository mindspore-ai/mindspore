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

export SLOG_PRINT_TO_STDOUT=1
source /root/miniconda3/bin/activate ci3.6
export RANK_SIZE=4
export RANK_TABLE_FILE=../../rank_table_4p.json
export RANK_ID=$1
export DEVICE_ID=$1
export HCCL_FLAG=1
export DEPLOY_MODE=0
export AICPU_FLAG=1
export DUMP_OP=1
export PYTHONPATH=../../../../../../../../mindspore:/usr/local/HiAI/runtime/python3.6/site-packages/topi.egg/:/usr/local/HiAI/runtime/python3.6/site-packages/te.egg/:/usr/local/HiAI/runtime/ops/op_impl/built-in/ai_core/tbe/
export OPTION_EXEC_EXTERN_PLUGIN_PATH=/usr/local/HiAI/runtime/lib64/libhccl.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/libfe.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/libaicpu_plugin.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/libge_local_engine.so:/usr/local/HiAI/runtime/lib64/plugin/opskernel/librts_engine.so
export LD_LIBRARY_PATH=/usr/local/HiAI/runtime/lib64
export FE_FLAG=1
export PATH=/usr/local/HiAI/runtime/ccec_compiler/bin:$PATH
if [ $1 -eq 0 ];
then
    export DUMP_GE_GRAPH=true
    export ME_DRAW_GRAPH=1
fi

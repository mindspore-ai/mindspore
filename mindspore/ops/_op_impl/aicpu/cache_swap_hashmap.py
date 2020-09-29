# Copyright 2020 Huawei Technologies Co., Ltd
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

"""CacheSwapHashmap op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

cache_swap_hashmap_op_info = AiCPURegOp("CacheSwapHashmap") \
    .fusion_type("OPAQUE") \
    .input(0, "hashmap", "required") \
    .input(1, "miss_emb_idx", "required") \
    .input(2, "step", "required") \
    .output(0, "swap_cache_idx", "required") \
    .output(1, "old_emb_idx", "required") \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, \
    DataType.I32_Default, DataType.I32_Default, \
    DataType.I32_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, \
    DataType.I32_Default, DataType.I64_Default, \
    DataType.I64_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, \
    DataType.I64_Default, DataType.I32_Default, \
    DataType.I32_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, \
    DataType.I64_Default, DataType.I64_Default, \
    DataType.I64_Default) \
    .get_op_info()

@op_info_register(cache_swap_hashmap_op_info)
def _cache_swap_hashmap_aicpu():
    """CacheSwapHashmap AiCPU register"""
    return

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

"""EmbeddingLookup op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

search_cache_idx_op_info = AiCPURegOp("SearchCacheIdx") \
    .fusion_type("OPAQUE") \
    .input(0, "hashmap", "required") \
    .input(1, "indices", "required") \
    .input(2, "step", "required") \
    .input(3, "emb_max_num", "required") \
    .input(4, "cache_max_num", "required") \
    .output(0, "cache_idx", "required") \
    .output(1, "miss_idx_1d", "required") \
    .output(2, "miss_emb_idx", "required") \
    .dtype_format(DataType.I32_Default, DataType.I32_Default,
                  DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.I32_Default, DataType.I32_Default,
                  DataType.I32_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default,
                  DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.I64_Default, DataType.I64_Default,
                  DataType.I64_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default,
                  DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.I32_Default, DataType.I32_Default,
                  DataType.I32_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default,
                  DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.I64_Default, DataType.I64_Default,
                  DataType.I64_Default) \
    .get_op_info()


@op_info_register(search_cache_idx_op_info)
def _search_cache_idx_aicpu():
    """SearchCacheIdx AiCPU register"""
    return

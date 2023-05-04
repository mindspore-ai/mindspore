# Copyright 2023 Huawei Technologies Co., Ltd
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

"""MemSet op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp

mem_set_info = TBERegOp("MemSet") \
    .fusion_type("ELEMWISE") \
    .async_flag(False) \
    .binfile_name("mem_set.so") \
    .compute_cost(10) \
    .kernel_name("mem_set") \
    .partial_flag(True) \
    .dynamic_compile_static(True) \
    .dynamic_shape(True) \
    .attr("sizes", "required", "listInt64", "all") \
    .attr("dtypes", "optional", "listInt", "all", "[]") \
    .attr("values_int", "optional", "listInt", "all", "[]") \
    .attr("values_float", "optional", "listFloat", "all", "[]") \
    .get_op_info()


@op_info_register(mem_set_info)
def _mem_set_tbe():
    """MemSet TBE register"""
    return

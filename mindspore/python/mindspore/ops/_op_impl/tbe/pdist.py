# Copyright 2022 Huawei Technologies Co., Ltd
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

"""Pdist op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

pdist_op_info = TBERegOp("Pdist") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("pdist.so") \
    .compute_cost(10) \
    .kernel_name("pdist") \
    .partial_flag(True) \
    .attr("p", "optional", "float", "all", "2.0") \
    .input(0, "x", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(pdist_op_info)
def _pdist_tbe():
    """Pdist TBE register"""
    return

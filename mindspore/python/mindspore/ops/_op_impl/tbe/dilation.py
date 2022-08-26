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

"""Dilation op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

dilation_op_info = TBERegOp("Dilation") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("dilation.so") \
    .compute_cost(10) \
    .kernel_name("dilation") \
    .partial_flag(True) \
    .attr("dilations", "required", "listInt", "all") \
    .attr("pads", "optional", "listInt", "all", "[]") \
    .attr("padding_value", "optional", "float", "all", "0.0") \
    .input(0, "x", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.I8_5HD, DataType.I8_5HD) \
    .dtype_format(DataType.F16_5HD, DataType.F16_5HD) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD) \
    .get_op_info()


@op_info_register(dilation_op_info)
def _dilation_tbe():
    """Dilation TBE register"""
    return

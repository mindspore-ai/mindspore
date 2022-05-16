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

"""pooling op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

pooling_op_info = TBERegOp("Pooling") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("pooling.so") \
    .compute_cost(10) \
    .kernel_name("pooling") \
    .partial_flag(True) \
    .attr("window", "optional", "listInt", "all", "1,1") \
    .attr("stride", "optional", "listInt", "all", "1,1") \
    .attr("offset_x", "optional", "int", "all", "0") \
    .attr("mode", "optional", "int", "all", "0") \
    .attr("pad", "optional", "listInt", "all", "0,0,0,0") \
    .attr("global_pooling", "optional", "bool", "all", "false") \
    .attr("ceil_mode", "optional", "int", "all", "0") \
    .attr("dilation", "optional", "listInt", "all", "1,1,1,1") \
    .input(0, "x", False, "required", "all") \
    .input(1, "matrix", False, "optional", "all") \
    .input(2, "bias", False, "optional", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_5HD, DataType.F16_FracZ, DataType.F16_Default, DataType.F16_5HD) \
    .dtype_format(DataType.I8_5HD, DataType.I8_FracZ, DataType.I32_Default, DataType.I32_5HD) \
    .get_op_info()


@op_info_register(pooling_op_info)
def _pooling_tbe():
    """Pooling TBE register"""
    return

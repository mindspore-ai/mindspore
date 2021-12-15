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

"""HSigmoid op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

hsigmoid_op_info = TBERegOp("HSigmoid") \
    .fusion_type("ELEMWISE") \
    .async_flag(False) \
    .binfile_name("hard_sigmoid.so") \
    .compute_cost(10) \
    .kernel_name("hard_sigmoid") \
    .partial_flag(True) \
    .attr("alpha", "optional", "float", "all", "0.16666666") \
    .attr("beta", "optional", "float", "all", "0.5") \
    .input(0, "input_x", False, "required", "all") \
    .output(0, "output_y", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F16_5HD, DataType.F16_5HD) \
    .dtype_format(DataType.F16_NHWC, DataType.F16_NHWC) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD) \
    .dtype_format(DataType.F32_NHWC, DataType.F32_NHWC) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default) \
    .dtype_format(DataType.I32_5HD, DataType.I32_5HD) \
    .dtype_format(DataType.I32_NHWC, DataType.I32_NHWC) \
    .get_op_info()


@op_info_register(hsigmoid_op_info)
def _hsigmoid_tbe():
    """HSigmoid TBE register"""
    return

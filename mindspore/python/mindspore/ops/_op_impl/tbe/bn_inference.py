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

"""BNInference op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

bn_inference_op_info = TBERegOp("BNInference") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("bninference_d.so") \
    .compute_cost(10) \
    .kernel_name("bninference_d") \
    .partial_flag(True) \
    .attr("momentum", "optional", "float", "all", "0.999") \
    .attr("epsilon", "optional", "float", "all", "0.00001") \
    .attr("use_global_stats", "optional", "bool", "true,false", "true") \
    .attr("mode", "optional", "int", "all", "1") \
    .input(0, "x", False, "required", "all") \
    .input(1, "mean", False, "required", "all") \
    .input(2, "variance", False, "required", "all") \
    .input(3, "scale", False, "optional", "all") \
    .input(4, "offset", False, "optional", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_5HD, DataType.F16_5HD, DataType.F16_5HD, DataType.F16_5HD,
                  DataType.F16_5HD, DataType.F16_5HD) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD,
                  DataType.F32_5HD, DataType.F32_5HD) \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(bn_inference_op_info)
def _bn_inference_tbe():
    """BNInference TBE register"""
    return

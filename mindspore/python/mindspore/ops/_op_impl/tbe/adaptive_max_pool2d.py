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

"""AdaptiveMaxPool2D op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

adaptive_max_pool2d_op_info = TBERegOp("AdaptiveMaxPool2D") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("adaptive_max_pool2d.so") \
    .compute_cost(10) \
    .kernel_name("adaptive_max_pool2d") \
    .partial_flag(True) \
    .attr("output_size", "required", "listInt", "all") \
    .input(0, "x", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .output(1, "argmax", False, "optional", "all") \
    .dtype_format(DataType.F16_5HD, DataType.F16_5HD, DataType.I32_5HD) \
    .get_op_info()


@op_info_register(adaptive_max_pool2d_op_info)
def _adaptive_max_pool2d_tbe():
    """AdaptiveMaxPool2d TBE register"""
    return

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

"""SoftmaxV2WithDropoutDoMaskV3 op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

softmax_v2_with_drop_out_do_mask_v3_op_info = TBERegOp("SoftmaxV2WithDropoutDoMaskV3") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("softmax_v2_with_drop_out_do_mask_v3_d.so") \
    .compute_cost(10) \
    .kernel_name("softmax_v2_with_drop_out_do_mask_v3_d") \
    .partial_flag(True) \
    .attr("keep_prob", "required", "float", "all") \
    .attr("axis", "optional", "listInt", "all", "-1") \
    .input(0, "x", False, "required", "all") \
    .input(1, "mask", False, "required", "all") \
    .output(0, "y1", False, "required", "all") \
    .output(1, "y2", False, "required", "all") \
    .dtype_format(DataType.F16_FracNZ, DataType.U8_Default, DataType.F16_FracNZ, DataType.F16_FracNZ) \
    .get_op_info()


@op_info_register(softmax_v2_with_drop_out_do_mask_v3_op_info)
def _softmax_v2_with_dropout_do_mask_v3_tbe():
    """SoftmaxV2WithDropoutDoMaskV3 TBE register"""
    return

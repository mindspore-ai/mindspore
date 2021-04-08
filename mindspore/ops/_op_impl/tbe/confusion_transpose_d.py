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

"""ConfusionTransposeD op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

confusion_transpose_d_op_info = TBERegOp("ConfusionTransposeD") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("confusion_transpose_d.so") \
    .compute_cost(10) \
    .kernel_name("confusion_transpose_d") \
    .partial_flag(True) \
    .attr("perm", "required", "listInt", "all") \
    .attr("shape", "required", "listInt", "all") \
    .attr("transpose_first", "required", "bool", "all") \
    .input(0, "x", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .is_dynamic_format(True) \
    .dtype_format(DataType.None_None, DataType.None_None) \
    .get_op_info()


@op_info_register(confusion_transpose_d_op_info)
def _confusion_transpose_d_tbe():
    """ConfusionTransposeD TBE register"""
    return

# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed unde:q!r the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""BCEWithLogitsLoss op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

bce_with_logits_loss_op_info = TBERegOp("BCEWithLogitsLoss") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("sigmoid_cross_entropy_with_logits_v2.so") \
    .compute_cost(10) \
    .kernel_name("sigmoid_cross_entropy_with_logits_v2") \
    .partial_flag(True) \
    .dynamic_shape(True) \
    .is_dynamic_format(True) \
    .attr("reduction", "optional", "str", "all", "mean") \
    .input(0, "predict", False, "required", "all") \
    .input(1, "target", False, "required", "all") \
    .input(2, "weight", False, "optional", "all") \
    .input(3, "pos_weight", False, "optional", "all") \
    .output(0, "loss", False, "required", "all") \
    .dtype_format(DataType.None_None, DataType.None_None, DataType.None_None, DataType.None_None,
                  DataType.None_None) \
    .get_op_info()


@op_info_register(bce_with_logits_loss_op_info)
def _bce_with_logits_loss_op_tbe():
    """BCEWithLogitsLoss TBE register"""
    return

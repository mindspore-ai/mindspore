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

"""KLDivLoss op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

kl_div_loss_op_info = TBERegOp("KLDivLoss") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("kl_div.so") \
    .compute_cost(10) \
    .kernel_name("kl_div") \
    .partial_flag(True) \
    .attr("reduction", "required", "str", "all") \
    .input(0, "x", False, "required", "all") \
    .input(1, "target", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .is_dynamic_format(True) \
    .dtype_format(DataType.None_None, DataType.None_None, DataType.None_None) \
    .get_op_info()


@op_info_register(kl_div_loss_op_info)
def _kl_div_loss_tbe():
    """KLDivLoss TBE register"""
    return

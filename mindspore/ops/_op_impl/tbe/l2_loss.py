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

"""L2Loss op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

l2_loss_op_info = TBERegOp("L2Loss") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("l2_loss.so") \
    .compute_cost(10) \
    .kernel_name("l2_loss") \
    .partial_flag(True) \
    .input(0, "x", None, "required", None) \
    .output(0, "y", True, "required", "all") \
    .is_dynamic_format(True) \
    .dtype_format(DataType.None_None, DataType.None_None) \
    .get_op_info()


@op_info_register(l2_loss_op_info)
def _l2_loss_tbe():
    """L2Loss TBE register"""
    return

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

"""BinaryCrossEntropy op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

binary_cross_entropy_ds_op_info = TBERegOp("BinaryCrossEntropy") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("binary_cross_entropy.so") \
    .compute_cost(10) \
    .kernel_name("binary_cross_entropy") \
    .partial_flag(True) \
    .dynamic_shape(True) \
    .attr("reduction", "optional", "str", "all") \
    .input(0, "x", False, "required", "all") \
    .input(1, "y", False, "required", "all") \
    .input(2, "weight", False, "optional", "all") \
    .output(0, "output", False, "required", "all") \
    .is_dynamic_format(True) \
    .dtype_format(DataType.None_None, DataType.None_None, DataType.None_None, DataType.None_None) \
    .get_op_info()


@op_info_register(binary_cross_entropy_ds_op_info)
def _binary_cross_entropy_ds_tbe():
    """BinaryCrossEntropy TBE register"""
    return

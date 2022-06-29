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

"""SparseApplyAdagrad op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

sparse_apply_adagrad_d_ds_op_info = TBERegOp("SparseApplyAdagrad") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("sparse_apply_adagrad_d.so") \
    .compute_cost(10) \
    .kernel_name("sparse_apply_adagrad_d") \
    .partial_flag(True) \
    .dynamic_shape(True) \
    .attr("lr", "required", "float", "all") \
    .attr("update_slots", "optional", "bool", "all") \
    .attr("use_locking", "optional", "bool", "all") \
    .input(0, "var", False, "required", "all") \
    .input(1, "accum", False, "required", "all") \
    .input(2, "grad", False, "required", "all") \
    .input(3, "indices", False, "required", "all") \
    .output(0, "var", False, "required", "all") \
    .output(1, "accum", False, "required", "all") \
    .dtype_format(DataType.F32_NHWC, DataType.F32_NHWC, DataType.F32_NHWC, DataType.I32_NHWC,
                  DataType.F32_NHWC, DataType.F32_NHWC) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.I32_Default,
                  DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(sparse_apply_adagrad_d_ds_op_info)
def _sparse_apply_adagrad_ds():
    """SparseApplyAdagradD TBE register"""
    return

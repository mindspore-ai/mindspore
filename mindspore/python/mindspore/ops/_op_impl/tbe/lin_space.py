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

"""LinSpace op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

lin_space_op_info = TBERegOp("LinSpace") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("lin_space.so") \
    .compute_cost(10) \
    .kernel_name("lin_space_d") \
    .partial_flag(True) \
    .op_pattern("broadcast") \
    .input(0, "assist", False, "required", "all") \
    .input(1, "start", False, "required", "all") \
    .input(2, "stop", False, "required", "all") \
    .input(3, "num", False, "required", "all") \
    .output(0, "output", False, "required", "all") \
    .dtype_format(DataType.F32_None, DataType.F32_None, DataType.F32_None, DataType.I32_None,
                  DataType.F32_None,) \
    .get_op_info()


@op_info_register(lin_space_op_info)
def _lin_space_tbe():
    """LinSpace TBE register"""
    return

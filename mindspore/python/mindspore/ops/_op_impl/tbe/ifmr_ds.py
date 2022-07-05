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

"""IFMR op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

ifmr_ds_op_info = TBERegOp("IFMR") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("ifmr.so") \
    .compute_cost(10) \
    .kernel_name("ifmr") \
    .partial_flag(True) \
    .dynamic_shape(True) \
    .attr("min_percentile", "required", "float", "all") \
    .attr("max_percentile", "required", "float", "all") \
    .attr("search_range", "required", "listFloat", "all") \
    .attr("search_step", "required", "float", "all") \
    .attr("with_offset", "required", "bool", "all") \
    .input(0, "data", False, "required", "all") \
    .input(1, "data_min", False, "required", "all") \
    .input(2, "data_max", False, "required", "all") \
    .input(3, "cumsum", False, "required", "all") \
    .output(0, "scale", False, "required", "all") \
    .output(1, "offset", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.I32_Default,
                  DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.I32_Default,
                  DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(ifmr_ds_op_info)
def _ifmr_ds_tbe():
    """IFMR TBE register"""
    return

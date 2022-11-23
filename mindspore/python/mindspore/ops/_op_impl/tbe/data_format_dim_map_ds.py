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

"""DataFormatDimMap op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

data_format_dim_map_op_info = TBERegOp("DataFormatDimMap") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("data_format_dim_map.so") \
    .compute_cost(10) \
    .kernel_name("data_format_dim_map") \
    .partial_flag(True) \
    .dynamic_shape(True) \
    .dynamic_compile_static(True) \
    .attr("dst_format", "optional", "str", "all") \
    .attr("src_format", "optional", "str", "all") \
    .input(0, "x", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.I32_5HD, DataType.I32_5HD) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default) \
    .get_op_info()


@op_info_register(data_format_dim_map_op_info)
def _data_format_dim_map_ds_tbe():
    """DataFormatDimMap TBE register"""
    return

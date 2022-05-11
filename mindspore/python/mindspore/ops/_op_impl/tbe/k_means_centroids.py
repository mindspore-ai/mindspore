# Copyright 2022-2022 Huawei Technologies Co., Ltd
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

"""k_means_centroids op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

k_means_centroids_op_info = TBERegOp("KMeansCentroids") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("k_means_centroids.so") \
    .compute_cost(10) \
    .kernel_name("k_means_centroids") \
    .partial_flag(True) \
    .attr("use_actual_distance", "required", "bool", "all", "false") \
    .input(0, "x", False, "required", "all") \
    .input(1, "y", False, "required", "all") \
    .input(2, "sum_square_y", False, "required", "all") \
    .input(3, "sum_square_x", False, "required", "all") \
    .output(0, "segment_sum", False, "required", "all") \
    .output(1, "segment_count", False, "required", "all") \
    .output(2, "kmean_total_sum", False, "required", "all") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
        DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default)\
    .get_op_info()


@op_info_register(k_means_centroids_op_info)
def _k_means_centroids_tbe():
    """KMeansCentroids TBE register"""
    return

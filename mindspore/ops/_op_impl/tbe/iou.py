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

"""Iou op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

iou_op_info = TBERegOp("IOU") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("iou.so") \
    .compute_cost(10) \
    .kernel_name("iou") \
    .partial_flag(True) \
    .attr("mode", "optional", "str", "all", "iou") \
    .attr("eps", "optional", "float", "all", "1.0") \
    .input(0, "bboxes", False, "required", "all") \
    .input(1, "gtboxes", False, "required", "all") \
    .output(0, "overlap", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
    .get_op_info()


@op_info_register(iou_op_info)
def _iou_tbe():
    """Iou TBE register"""
    return

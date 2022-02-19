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

"""CropAndResizeGradBoxes op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType
crop_and_resize_grad_boxes_op_info = AiCPURegOp("CropAndResizeGradBoxes") \
    .fusion_type("OPAQUE") \
    .input(0, "grads", "required") \
    .input(1, "images", "required") \
    .input(2, "boxes", "required") \
    .input(3, "box_index", "required") \
    .output(0, "y", "required") \
    .attr("method", "str", "bilinear") \
    .dtype_format(DataType.F32_Default, DataType.U8_Default, DataType.F32_Default, DataType.I32_Default,
                  DataType.F32_Default) \
    .dtype_format(DataType.F32_Default, DataType.U16_Default, DataType.F32_Default, DataType.I32_Default,
                  DataType.F32_Default) \
    .dtype_format(DataType.F32_Default, DataType.I8_Default, DataType.F32_Default, DataType.I32_Default,
                  DataType.F32_Default) \
    .dtype_format(DataType.F32_Default, DataType.I16_Default, DataType.F32_Default, DataType.I32_Default,
                  DataType.F32_Default) \
    .dtype_format(DataType.F32_Default, DataType.I32_Default, DataType.F32_Default, DataType.I32_Default,
                  DataType.F32_Default) \
    .dtype_format(DataType.F32_Default, DataType.I64_Default, DataType.F32_Default, DataType.I32_Default,
                  DataType.F32_Default) \
    .dtype_format(DataType.F32_Default, DataType.F16_Default, DataType.F32_Default, DataType.I32_Default,
                  DataType.F32_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.I32_Default,
                  DataType.F32_Default) \
    .dtype_format(DataType.F32_Default, DataType.F64_Default, DataType.F32_Default, DataType.I32_Default,
                  DataType.F32_Default) \
    .dtype_format(DataType.F32_NHWC, DataType.U8_NHWC, DataType.F32_NHWC, DataType.I32_NHWC,
                  DataType.F32_NHWC) \
    .dtype_format(DataType.F32_NHWC, DataType.U16_NHWC, DataType.F32_NHWC, DataType.I32_NHWC,
                  DataType.F32_NHWC) \
    .dtype_format(DataType.F32_NHWC, DataType.I8_NHWC, DataType.F32_NHWC, DataType.I32_NHWC,
                  DataType.F32_NHWC) \
    .dtype_format(DataType.F32_NHWC, DataType.I16_NHWC, DataType.F32_NHWC, DataType.I32_NHWC,
                  DataType.F32_NHWC) \
    .dtype_format(DataType.F32_NHWC, DataType.I32_NHWC, DataType.F32_NHWC, DataType.I32_NHWC,
                  DataType.F32_NHWC) \
    .dtype_format(DataType.F32_NHWC, DataType.I64_NHWC, DataType.F32_NHWC, DataType.I32_NHWC,
                  DataType.F32_NHWC) \
    .dtype_format(DataType.F32_NHWC, DataType.F16_NHWC, DataType.F32_NHWC, DataType.I32_NHWC,
                  DataType.F32_NHWC) \
    .dtype_format(DataType.F32_NHWC, DataType.F32_NHWC, DataType.F32_NHWC, DataType.I32_NHWC,
                  DataType.F32_NHWC) \
    .dtype_format(DataType.F32_NHWC, DataType.F64_NHWC, DataType.F32_NHWC, DataType.I32_NHWC,
                  DataType.F32_NHWC) \
    .get_op_info()


@op_info_register(crop_and_resize_grad_boxes_op_info)
def _crop_and_resize_grad_boxes_aicpu():
    """CropAndResizeGradBoxes AiCPU register"""
    return

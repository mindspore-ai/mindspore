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

"""CropAndResize op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType
crop_and_resize_op_info = AiCPURegOp("CropAndResize") \
    .fusion_type("OPAQUE") \
    .input(0, "image", "required") \
    .input(1, "boxes", "required") \
    .input(2, "box_index", "required") \
    .input(3, "crop_size", "required") \
    .output(0, "y", "required") \
    .attr("method", "str") \
    .attr("extrapolation_value", "float") \
    .dtype_format(DataType.I8_Default, DataType.F32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.F32_Default) \
    .dtype_format(DataType.I16_Default, DataType.F32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.F32_Default) \
    .dtype_format(DataType.I32_Default, DataType.F32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.F32_Default) \
    .dtype_format(DataType.I64_Default, DataType.F32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.F32_Default) \
    .dtype_format(DataType.F16_Default, DataType.F32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.F32_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.F32_Default) \
    .dtype_format(DataType.F64_Default, DataType.F32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.F32_Default) \
    .dtype_format(DataType.U8_Default, DataType.F32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.F32_Default) \
    .dtype_format(DataType.U16_Default, DataType.F32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.F32_Default) \
    .dtype_format(DataType.I8_NHWC, DataType.F32_NHWC, DataType.I32_NHWC, DataType.I32_NHWC,
                  DataType.F32_NHWC) \
    .dtype_format(DataType.I16_NHWC, DataType.F32_NHWC, DataType.I32_NHWC, DataType.I32_NHWC,
                  DataType.F32_NHWC) \
    .dtype_format(DataType.I32_NHWC, DataType.F32_NHWC, DataType.I32_NHWC, DataType.I32_NHWC,
                  DataType.F32_NHWC) \
    .dtype_format(DataType.I64_NHWC, DataType.F32_NHWC, DataType.I32_NHWC, DataType.I32_NHWC,
                  DataType.F32_NHWC) \
    .dtype_format(DataType.F16_NHWC, DataType.F32_NHWC, DataType.I32_NHWC, DataType.I32_NHWC,
                  DataType.F32_NHWC) \
    .dtype_format(DataType.F32_NHWC, DataType.F32_NHWC, DataType.I32_NHWC, DataType.I32_NHWC,
                  DataType.F32_NHWC) \
    .dtype_format(DataType.F64_NHWC, DataType.F32_NHWC, DataType.I32_NHWC, DataType.I32_NHWC,
                  DataType.F32_NHWC) \
    .dtype_format(DataType.U8_NHWC, DataType.F32_NHWC, DataType.I32_NHWC, DataType.I32_NHWC,
                  DataType.F32_NHWC) \
    .dtype_format(DataType.U16_NHWC, DataType.F32_NHWC, DataType.I32_NHWC, DataType.I32_NHWC,
                  DataType.F32_NHWC) \
    .get_op_info()


@op_info_register(crop_and_resize_op_info)
def _crop_and_resize_aicpu():
    """CropAndResize AiCPU register"""
    return

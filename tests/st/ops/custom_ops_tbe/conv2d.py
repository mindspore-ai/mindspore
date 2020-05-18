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
from __future__ import absolute_import
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from .conv_layer import conv_layer_cce
from .conv_layer_fast import conv_layer_fast_cce
from topi.cce import util
from te import platform as cce

Nonetype = type(None)


# pylint: disable=unused-argument, no-value-for-parameter, too-many-branches
@fusion_manager.register("conv2d")
def conv2d_compute(inputs, weights, bias, outputs, strides, pad_list, dilations,
                   kernel_name="conv2d"):
    """
    conv2d compute

    Notice
    ------
    only used by framework combine with IR

    Parameters
    ----------
    inputs: tvm placeholder
        input 5hd feature map tensor
    weights: tvm placeholder
        input frac_z weight tensor
    outputs: tvm placeholder
        output tensor, dtype must be assigned
    bias: tvm placeholder or None
        input 1d bias tensor
    strides: integers
        stride on H/W, format sensitive
    pads: tuple/list of 4 integers
        [pad_top, pad_bottom, pad_left, pad_right]
    dilations: integers
        dilation on H/W, format sensitive
    kernel_name: string
        kernel name, default value is "conv2d"

    Returns
    -------
    tvm compute
    """
    shape_w = []
    for i in weights.op.attrs['ori_shape']:
        shape_w.append(i.value)

    format_w = weights.op.attrs['ori_format']
    if format_w == "NCHW":
        weight_h = shape_w[2]
        weight_w = shape_w[3]
    elif format_w == "NHWC":
        weight_h = shape_w[1]
        weight_w = shape_w[2]
    elif format_w == "HWCN":
        weight_h = shape_w[0]
        weight_w = shape_w[1]
    else:
        raise RuntimeError("weights ori_format should be NCHW, NHWC or HWCN")

    format_x = inputs.op.attrs['ori_format']
    if format_x == "NCHW":
        strideh = strides[0]
        stridew = strides[0]
        dlt_h = dilations[0]
        dlt_w = dilations[0]
    elif format_x == "NHWC":
        strideh = strides[0]
        stridew = strides[0]
        dlt_h = dilations[0]
        dlt_w = dilations[0]
    else:
        raise RuntimeError("inputs ori_format should be NCHW or NHWC")

    if len(pad_list) == 4:
        padh = [pad_list[0], pad_list[1]]
        padw = [pad_list[2], pad_list[3]]
    else:
        raise RuntimeError("pads shape should be 4d.")

    para_dict = {"pad_h": padh, "pad_w": padw, "stride_h": strideh, "stride_w": stridew,
                 "filter_h": weight_h, "filter_w": weight_w, "bias_tensor": bias}

    if cce.CceProductParams().cce_product == "5.10":
        para_dict["mad_dtype"] = "float16"
        res = te.lang.cce.conv(inputs, weights, para_dict)
    else:
        res = te.lang.cce.conv(inputs, weights, para_dict)

    return res


@util.check_input_type(dict, dict, (dict, Nonetype), dict, (tuple, list), (tuple, list), (tuple, list),
                       str)
def conv2d(inputs, weights, bias, outputs, strides, pad_list, dilations,
           kernel_name="conv2d"):
    """
    algorithm: conv2d

    Notice
    ------
    only used by framework combine with IR

    Parameters
    ----------
    inputs: dict with keys(shape and dtype)
        input 4d feature map tensor
    weights: dict with keys(shape and dtype)
        input 4d weight tensor
    outputs: dict with keys(shape and dtype)
        output tensor, dtype must be assigned
    bias: dict with keys(shape and dtype) or None
        input bias tensor
    strides: integers
        stride on H/W, format sensitive
    pads: integers
        [pad_top, pad_bottom, pad_left, pad_right]
    dilations: tuple/list of 4 integers
        dilation on H/W, format sensitive
    kernel_name: str
        kernel name, default value is "conv2d"

    Returns
    -------
    None
    """
    shape_x = inputs.get("ori_shape")
    in_dtype = inputs.get("dtype")
    shape_w = weights.get("ori_shape")
    w_dtype = weights.get("dtype")
    res_dtype = outputs.get("dtype")

    if len(pad_list) == 4:
        padh = [pad_list[0], pad_list[1]]
        padw = [pad_list[2], pad_list[3]]
    else:
        raise RuntimeError("pads shape should be 4d.")

    if (not isinstance(shape_x, (tuple, list))) or len(shape_x) != 4:
        raise RuntimeError("inputs should be 4d list.")

    if (not isinstance(shape_w, (tuple, list))) or len(shape_w) != 4:
        raise RuntimeError("weights should be 4d list.")

    format_x = inputs.get("ori_format")
    if format_x == "NCHW":
        shape_fm = shape_x
        strideh = strides[0]
        stridew = strides[0]
        dlt_h = dilations[0]
        dlt_w = dilations[0]
    elif format_x == "NHWC":
        shape_fm = [shape_x[0], shape_x[3], shape_x[1], shape_x[2]]
        strideh = strides[0]
        stridew = strides[0]
        dlt_h = dilations[0]
        dlt_w = dilations[0]
    else:
        raise RuntimeError("inputs ori_format should be NCHW or NHWC.")

    format_w = weights.get("ori_format")
    if format_w == "NCHW":
        shape_filter = shape_w
    elif format_w == "NHWC":
        shape_filter = [shape_w[0], shape_w[3], shape_w[1], shape_w[2]]
    elif format_w == "HWCN":
        shape_filter = [shape_w[3], shape_w[2], shape_w[0], shape_w[1]]
    else:
        raise RuntimeError("weights ori_format should be NCHW, NHWC or HWCN.")

    if bias is None:
        use_bias = False
    else:
        use_bias = True

    if cce.CceProductParams().cce_product == "5.10":
        conv_layer_fast_cce(shape_fm, shape_filter, in_dtype, w_dtype, res_dtype,
                            padh, padw, strideh, stridew, bias=use_bias,
                            kernel_name=kernel_name, need_build=True, need_print=False)
    else:
        conv_layer_cce(shape_fm, shape_filter, in_dtype, w_dtype, res_dtype,
                       padh, padw, strideh, stridew,
                       quantize_config=[0, 0, 0], scale_sqrt=[0, 0, 0],
                       bias=use_bias, kernel_name=kernel_name,
                       need_build=True, need_print=False)

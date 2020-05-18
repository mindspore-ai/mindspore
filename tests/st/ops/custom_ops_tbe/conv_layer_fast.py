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
import te.lang.cce
from te import tvm
from te.platform import CUBE_MKN
from topi import generic
from topi.cce import util

# pylint: disable=R0913,R0914,R0915,E1101
# the dim of shape in conv must be 4
PAD_SHAPE_DIM = 2

NoneType = type(None)


@util.check_input_type((list, tuple), (list, tuple), str, str, str,
                       (list, int), (list, int), int, int, bool, str)
def conv_layer_fast_cce_para_check(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                   padh, padw, strideh, stridew, bias, kernel_name):
    # conv shape check
    util.check_kernel_name(kernel_name)

    # conv data type check
    util.check_dtype_rule(in_dtype, ['float16'])
    util.check_dtype_rule(w_dtype, ['float16'])
    util.check_dtype_rule(res_dtype, ['float16'])

    if not isinstance(bias, bool):
        raise RuntimeError("bias dtype should be bool.")

    if isinstance(padh, list):
        if len(padh) != PAD_SHAPE_DIM:
            raise RuntimeError("Dimension must be %d when padh is a list." % PAD_SHAPE_DIM)
        pad_top = padh[0]
        pad_bottom = padh[1]
    else:
        pad_top = padh
        pad_bottom = padh

    if isinstance(padw, list):
        if len(padw) != PAD_SHAPE_DIM:
            raise RuntimeError("Dimension must be %d when padw is a list." % PAD_SHAPE_DIM)
        pad_left = padw[0]
        pad_right = padw[1]
    else:
        pad_left = padw
        pad_right = padw

    shape_in, shape_w = te.lang.cce.check_conv_shape(shape_in, shape_w, pad_top, pad_bottom,
                                                     pad_left, pad_right, strideh, stridew,
                                                     in_dtype, w_dtype, res_dtype)

    return shape_in, shape_w


@util.check_input_type((list, tuple), (list, tuple), str, str, str,
                       (list, int), (list, int), int, int,
                       bool, str, bool, bool)
def conv_layer_fast_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                        padh, padw, strideh, stridew, bias=False,
                        kernel_name="cce_conv",
                        need_build=False, need_print=False):
    """

    Parameters
    ----------
    shape_in : shape of data_in

    shape_w : shape of filter

    in_dtype : the feature map data type

    w_dtype : the weight data type

    res_dtype : the result data type

    padh: the padding shape in H

    padw: the padding shape in weight

    strideh: the stride value in H

    stridew: the stride value in weight

    bias: the tag for bias or not

    kernel_name : cce kernel name, default value is "cce_conv"

    need_buid : if need to build CCEC kernel, default value is False

    need_print : if need to print the ir, default value is False

    Returns
    -------
    None

    """
    in_dtype = in_dtype.lower()
    w_dtype = w_dtype.lower()
    res_dtype = res_dtype.lower()

    shape_in = list(shape_in)
    shape_w = list(shape_w)

    shape_in, shape_w = conv_layer_fast_cce_para_check(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                                       padh, padw, strideh, stridew, bias, kernel_name)

    batch_size = shape_in[0]
    in_channel = shape_in[1]
    feature_map_h = shape_in[2]
    feature_map_w = shape_in[3]
    block_size_k = CUBE_MKN[in_dtype]['mac'][1]
    fmap_shape_nc1hwc0 = (batch_size, (in_channel + block_size_k - 1) // block_size_k,
                          feature_map_h, feature_map_w, block_size_k)

    out_channel = shape_w[0]
    in_channel_weight = shape_w[1]
    filter_h = shape_w[2]
    filter_w = shape_w[3]
    block_size_k = CUBE_MKN[w_dtype]['mac'][1]
    block_size_n = CUBE_MKN[w_dtype]['mac'][2]
    filter_shape_frac_z = (in_channel_weight * filter_h * filter_w // block_size_k,
                           out_channel // block_size_n, block_size_n, block_size_k)

    with tvm.target.cce():
        data = tvm.placeholder(
            fmap_shape_nc1hwc0, name='Fmap', dtype=in_dtype)
        weight = tvm.placeholder(
            filter_shape_frac_z, name='Filter', dtype=w_dtype)
        bias_tensor = None

        if bias:
            bias_tensor = tvm.placeholder(
                (out_channel,), name='bias_tensor', dtype=res_dtype)

        mad_dtype = "float16"

        conv_res = te.lang.cce.conv(
            data, weight, {"bias_tensor": bias_tensor,
                           "scale_q": None,
                           "offset_q": None,
                           "scale_drq": None,
                           "offset_pad": None,
                           "offset_rq": None,
                           "quantize_config": [0, 0, 0],
                           "is_quantize": False,
                           "is_dequantize": False,
                           "is_requantize": False,
                           "scale_sqrt": [0, 0, 0],
                           "pad_h": padh, "pad_w": padw,
                           "stride_h": strideh, "stride_w": stridew,
                           "filter_h": filter_h, "filter_w": filter_w,
                           "res_dtype": res_dtype, "mad_dtype": mad_dtype},
            dsl_flag=False)
        if bias:
            tensor_list = [data, weight, bias_tensor, conv_res]
        else:
            tensor_list = [data, weight, conv_res]
        sch = generic.auto_schedule(conv_res)

    config = {
        "print_ir": need_print,
        "need_build": need_build,
        "name": kernel_name,
        "tensor_list": tensor_list
    }

    te.lang.cce.cce_build_code(sch, config)

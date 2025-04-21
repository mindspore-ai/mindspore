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
from tbe.tvm.topi import generic
from tbe.tvm.topi.cce import util
from tbe.tvm.topi.cce.util import is_v200_version

# pylint: disable=R0912,R0913,R0914,R0915,E1101
# the dim of shape in conv must be 4
PAD_SHAPE_DIM = 2

NONETYPE = type(None)


@util.check_input_type((list, tuple), (list, tuple), str, str, str, (list, int), (list, int),
                       int, int, (list, tuple), (list, tuple),
                       str, str, str,
                       str, str, str,
                       str, bool, str)
def conv_layer_cce_para_check(shape_in, shape_w, in_dtype, w_dtype, res_dtype, padh, padw,
                              strideh, stridew, quantize_config, scale_sqrt,
                              scale_q_dtype, offset_q_dtype, scale_dq_dtype,
                              scale_rq_dtype, offset_rq_dtype, offset_w_dtype,
                              offset_pad_dtype, bias, kernel_name):
    # conv shape check
    util.check_kernel_name(kernel_name)

    # conv data type check
    util.check_dtype_rule(in_dtype, ['float16', 'int8', 'uint8'])
    util.check_dtype_rule(w_dtype, ['float16', 'int8', 'uint8'])
    res_dtype_list = ['float16', 'int8', 'uint8']
    if is_v200_version():
        res_dtype_list.append('int32')
    util.check_dtype_rule(res_dtype, res_dtype_list)
    util.check_dtype_rule(scale_q_dtype, ['float16'])
    util.check_dtype_rule(offset_q_dtype, ['float16'])
    util.check_dtype_rule(scale_dq_dtype, ['float16'])
    util.check_dtype_rule(scale_rq_dtype, ['float16'])
    util.check_dtype_rule(offset_rq_dtype, ['float16'])
    util.check_dtype_rule(offset_w_dtype, ['int32'])
    util.check_dtype_rule(offset_pad_dtype, ['uint8'])

    if not isinstance(bias, bool):
        raise RuntimeError("bias dtype should be bool.")

    if quantize_config[0] == 0:
        if is_v200_version():
            util.check_dtype_rule(in_dtype, ('int8',))
            util.check_dtype_rule(w_dtype, ('int8',))
            util.check_dtype_rule(res_dtype, ('int32',))
        else:
            util.check_dtype_rule(in_dtype, ['float16'])
            util.check_dtype_rule(w_dtype, ['float16'])
            util.check_dtype_rule(res_dtype, ['float16'])

    if quantize_config[0] == 1:
        util.check_dtype_rule(w_dtype, ['int8'])
        if quantize_config[1] == 0:
            util.check_dtype_rule(in_dtype, ['int8', 'float16'])
            util.check_dtype_rule(res_dtype, ['int8', 'float16'])
        elif quantize_config[1] == 1:
            util.check_dtype_rule(in_dtype, ['uint8', 'float16'])
            util.check_dtype_rule(res_dtype, ['uint8', 'float16'])
        elif quantize_config[1] == 2:
            raise RuntimeError("All Offset mode quantize not support.")
        else:
            raise RuntimeError("Invalid quantize algorithm.")

    # quantize switch on
    if quantize_config[0] == 1:
        # quantize -> DeQuantize dataflow
        if in_dtype == 'float16' and w_dtype == 'int8' and res_dtype == 'float16':
            pass
        # DeQuantize dataflow
        elif (in_dtype in ['int8', 'uint8'] and w_dtype == 'int8' and
              res_dtype == 'float16'):
            pass
        # quantize -> ReQuantize dataflow
        elif (in_dtype == 'float16' and w_dtype == 'int8' and res_dtype in
              ['int8', 'uint8']):
            pass
        # ReQuantize dataflow
        elif (in_dtype in ['int8', 'uint8'] and w_dtype == 'int8' and res_dtype in
              ['int8', 'uint8']):
            pass
        else:
            raise RuntimeError("Not support in/out data type for quantize.")

        if quantize_config not in ([1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]):
            raise RuntimeError("Invalid Quantize Config.")

        if scale_sqrt not in ([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1],
                              [1, 0, 1], [0, 1, 1], [1, 1, 1]):
            raise RuntimeError("Invalid Quantize Config.")

    # quantize switch off
    elif quantize_config[0] == 0:
        if quantize_config != [0, 0, 0]:
            raise RuntimeError("Invalid Quantize Config.")
        if scale_sqrt != [0, 0, 0]:
            raise RuntimeError("Invalid Quantize Config.")
    else:
        raise RuntimeError("Invalid Quantize Config.")

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

    shape_in, shape_w = te.lang.cce.check_conv_shape(shape_in, shape_w, pad_top, pad_bottom, \
                                                     pad_left, pad_right, strideh, \
                                                     stridew, in_dtype, w_dtype, res_dtype)

    return shape_in, shape_w


@util.check_input_type((list, tuple), (list, tuple), str, str, str, \
                       (list, int), (list, int), int, int,
                       (list, NONETYPE), (list, NONETYPE),
                       str, str, str,
                       str, str, str, str,
                       bool, str, bool, bool)
def conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype, padh, padw, strideh, stridew,
                   quantize_config=None, scale_sqrt=None,
                   scale_q_dtype='float16', offset_q_dtype='float16', scale_dq_dtype='float16',
                   scale_rq_dtype='float16', offset_rq_dtype='float16', offset_w_dtype='int32',
                   offset_pad_dtype='uint8', bias=False, kernel_name="cce_conv", need_build=False,
                   need_print=False):
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

    quantize_config: quantize config table, default [0, 0, 0]
    quantize_config[0] - quantize function switch
                        0: quantize off
                        1: quantize on
    quantize_config[1] - quantize_algorithm
                        0: non offset
                        1: half offset
                        2: all offset ( Not supported now )
    quantize_config[2] - QuantizeScaleType (for Dequantize/Requantize, quantize always scalar)
                        0: scalar
                        1: vector

    scale_sqrt: scale mode
    scale_sqrt[0] - Quantize scale mode
                   0: non sqrt
                   1: sqrt
    scale_sqrt[1] - DeQuantize scale mode
                   0: non sqrt
                   1: sqrt
    scale_sqrt[2] - ReQuantize scale mode
                   0: non sqrt
                   1: sqrt

    scale_q_dtype: Quantize scale data type, default 'float16'

    offset_q_dtype: Quantize offset data type, default 'float16'

    scale_dq_dtype: DeQuantize scale data type, default 'float16'

    scale_rq_dtype: ReQuantize scale data type, default 'float16'

    offset_rq_dtype: ReQuantize offset data type, default 'float16'

    offset_w_dtype: weight offset data type, default 'int32'

    offset_pad_dtype: Quantize Cube offset data type, default 'uint8'

    bias: the tag for bias or not

    kernel_name : cce kernel name, default value is "cce_conv"

    need_build : if need to build CCEC kernel, default value is False

    need_print : if need to print the ir, default value is False

    Returns
    -------
    wrapped_tensor

    """
    # for pylint, otherwise "Dangerous default value [] as argument"
    if quantize_config is None:
        quantize_config = [0, 0, 0]
    if scale_sqrt is None:
        scale_sqrt = [0, 0, 0]

    in_dtype = in_dtype.lower()
    w_dtype = w_dtype.lower()
    res_dtype = res_dtype.lower()
    scale_q_dtype = scale_q_dtype.lower()
    offset_q_dtype = offset_q_dtype.lower()
    scale_dq_dtype = scale_dq_dtype.lower()
    scale_rq_dtype = scale_rq_dtype.lower()
    offset_rq_dtype = offset_rq_dtype.lower()
    offset_w_dtype = offset_w_dtype.lower()
    offset_pad_dtype = offset_pad_dtype.lower()

    mad_dtype = 'float32'
    if w_dtype == 'int8':
        mad_dtype = 'int32'

    shape_in = list(shape_in)
    shape_w = list(shape_w)

    shape_in, shape_w = conv_layer_cce_para_check(shape_in, shape_w, in_dtype, w_dtype, res_dtype, padh, padw, strideh,
                                                  stridew,
                                                  quantize_config, scale_sqrt, scale_q_dtype, offset_q_dtype,
                                                  scale_dq_dtype,
                                                  scale_rq_dtype, offset_rq_dtype, offset_w_dtype, offset_pad_dtype,
                                                  bias, kernel_name)

    # quantize switch on
    if quantize_config[0] == 1:
        quantize_turn_on = True
        # quantize -> DeQuantize dataflow
        if in_dtype == 'float16' and w_dtype == 'int8' and res_dtype == 'float16':
            is_quantize = True
            is_dequantize = True
            is_requantize = False
        # DeQuantize dataflow
        elif (in_dtype in ['int8', 'uint8'] and w_dtype == 'int8' and
              res_dtype == 'float16'):
            is_quantize = False
            is_dequantize = True
            is_requantize = False
        # quantize -> ReQuantize dataflow
        elif (in_dtype == 'float16' and w_dtype == 'int8' and res_dtype in
              ['int8', 'uint8']):
            is_quantize = True
            is_dequantize = False
            is_requantize = True
        # ReQuantize dataflow
        elif (in_dtype in ['int8', 'uint8'] and w_dtype == 'int8' and res_dtype in
              ['int8', 'uint8']):
            is_quantize = False
            is_dequantize = False
            is_requantize = True
        else:
            raise RuntimeError("Not support in/out data type for quantize.")

    # quantize switch off
    elif quantize_config[0] == 0:
        quantize_turn_on = False
        is_quantize = False
        is_dequantize = False
        is_requantize = False

        if quantize_config != [0, 0, 0]:
            raise RuntimeError("Invalid Quantize Config.")
        if scale_sqrt != [0, 0, 0]:
            raise RuntimeError("Invalid Quantize Config.")
    else:
        raise RuntimeError("Invalid Quantize Config.")

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
        scale_q = None
        scale_dq = None
        scale_rq = None
        offset_pad = None
        offset_rq = None
        offset_q = None
        scale_drq = None

        # bias or fusion_bias(half offset)
        if bias or (quantize_config[1] == 1 and quantize_turn_on):
            bias_tensor = tvm.placeholder(
                (out_channel,), name='bias_tensor', \
                dtype="int32" if quantize_turn_on else res_dtype)

        # quantize on
        if quantize_turn_on:
            quantize_algorithm = quantize_config[1]
            if is_quantize:
                scale_q = tvm.placeholder(
                    (CUBE_MKN[scale_q_dtype]['mac'][1],), name='scaleQ', dtype=scale_q_dtype)
                if quantize_algorithm == 1:
                    offset_q = tvm.placeholder(
                        (CUBE_MKN[offset_q_dtype]['mac'][1],), name='offsetQ', dtype=offset_q_dtype)

            if is_dequantize:
                scale_dq_shape = (CUBE_MKN[scale_dq_dtype]['mac'][1],) if quantize_config[2] == 0 \
                    else (out_channel,)
                scale_dq = tvm.placeholder(
                    scale_dq_shape, name='scaleDq', dtype=scale_dq_dtype)

            if is_requantize:
                scale_rq_shape = (CUBE_MKN[scale_rq_dtype]['mac'][1],) if quantize_config[2] == 0 \
                    else (out_channel,)
                scale_rq = tvm.placeholder(
                    scale_rq_shape, name='scaleRq', dtype=scale_rq_dtype)
                if quantize_algorithm == 1:
                    offset_rq_shape = (CUBE_MKN[offset_rq_dtype]['mac'][1],)
                    offset_rq = tvm.placeholder(
                        offset_rq_shape, name='offsetRq', dtype=offset_rq_dtype)

            # need offset_pad , for half offset
            if quantize_algorithm == 1:
                offset_pad = tvm.placeholder(
                    (CUBE_MKN[offset_pad_dtype]['mac'][1],), name='offset_pad',
                    dtype=offset_pad_dtype)

            if quantize_algorithm == 0:
                if is_quantize:
                    if is_dequantize:
                        scale_drq = scale_dq
                    else:
                        scale_drq = scale_rq

                    conv_res = te.lang.cce.conv(
                        data, weight, {"bias_tensor": bias_tensor,
                                       "scale_q": scale_q,
                                       "offset_q": offset_q,
                                       "scale_drq": scale_drq,
                                       "offset_pad": offset_pad,
                                       "offset_rq": offset_rq,
                                       "quantize_config": quantize_config,
                                       "is_quantize": is_quantize,
                                       "is_dequantize": is_dequantize,
                                       "is_requantize": is_requantize,
                                       "scale_sqrt": scale_sqrt,
                                       "pad_h": padh, "pad_w": padw,
                                       "stride_h": strideh, "stride_w": stridew,
                                       "filter_h": filter_h, "filter_w": filter_w,
                                       "res_dtype": res_dtype, "mad_dtype": mad_dtype},
                        dsl_flag=False)
                    if bias:
                        tensor_list = [data, weight, bias_tensor, scale_q,
                                       scale_drq, conv_res]
                    else:
                        tensor_list = [data, weight, scale_q,
                                       scale_drq, conv_res]
                else:
                    if is_dequantize:
                        scale_drq = scale_dq
                    else:
                        scale_drq = scale_rq
                    conv_res = te.lang.cce.conv(
                        data, weight, {"bias_tensor": bias_tensor,
                                       "scale_q": scale_q,
                                       "offset_q": offset_q,
                                       "scale_drq": scale_drq,
                                       "offset_pad": offset_pad,
                                       "offset_rq": offset_rq,
                                       "quantize_config": quantize_config,
                                       "is_quantize": is_quantize,
                                       "is_dequantize": is_dequantize,
                                       "is_requantize": is_requantize,
                                       "scale_sqrt": scale_sqrt,
                                       "pad_h": padh, "pad_w": padw,
                                       "stride_h": strideh, "stride_w": stridew,
                                       "filter_h": filter_h, "filter_w": filter_w,
                                       "res_dtype": res_dtype, "mad_dtype": mad_dtype},
                        dsl_flag=False)
                    if bias:
                        tensor_list = [data, weight, bias_tensor,
                                       scale_drq, conv_res]
                    else:
                        tensor_list = [data, weight,
                                       scale_drq, conv_res]

            # half offset
            else:
                if is_quantize:
                    if is_dequantize:
                        scale_drq = scale_dq
                    else:
                        scale_drq = scale_rq
                    conv_res = te.lang.cce.conv(
                        data, weight, {"bias_tensor": bias_tensor,
                                       "scale_q": scale_q,
                                       "offset_q": offset_q,
                                       "scale_drq": scale_drq,
                                       "offset_pad": offset_pad,
                                       "offset_rq": offset_rq,
                                       "quantize_config": quantize_config,
                                       "is_quantize": is_quantize,
                                       "is_dequantize": is_dequantize,
                                       "is_requantize": is_requantize,
                                       "scale_sqrt": scale_sqrt,
                                       "pad_h": padh, "pad_w": padw,
                                       "stride_h": strideh, "stride_w": stridew,
                                       "filter_h": filter_h, "filter_w": filter_w,
                                       "res_dtype": res_dtype, "mad_dtype": mad_dtype},
                        dsl_flag=False)
                    if is_dequantize:
                        tensor_list = [data, weight, bias_tensor, scale_q, offset_q,
                                       scale_drq, offset_pad, conv_res]
                    else:
                        tensor_list = [data, weight, bias_tensor, scale_q, offset_q,
                                       scale_drq, offset_rq, offset_pad, conv_res]
                else:
                    if is_dequantize:
                        scale_drq = scale_dq
                    else:
                        scale_drq = scale_rq
                    conv_res = te.lang.cce.conv(
                        data, weight, {"bias_tensor": bias_tensor,
                                       "scale_q": scale_q,
                                       "offset_q": offset_q,
                                       "scale_drq": scale_drq,
                                       "offset_pad": offset_pad,
                                       "offset_rq": offset_rq,
                                       "quantize_config": quantize_config,
                                       "is_quantize": is_quantize,
                                       "is_dequantize": is_dequantize,
                                       "is_requantize": is_requantize,
                                       "scale_sqrt": scale_sqrt,
                                       "pad_h": padh, "pad_w": padw,
                                       "stride_h": strideh, "stride_w": stridew,
                                       "filter_h": filter_h, "filter_w": filter_w,
                                       "res_dtype": res_dtype, "mad_dtype": mad_dtype},
                        dsl_flag=False)
                    if is_dequantize:
                        tensor_list = [data, weight, bias_tensor,
                                       scale_drq, offset_pad, conv_res]
                    else:
                        tensor_list = [data, weight, bias_tensor,
                                       scale_drq, offset_rq, offset_pad, conv_res]
        else:
            conv_res = te.lang.cce.conv(
                data, weight, {"bias_tensor": bias_tensor,
                               "scale_q": scale_q,
                               "offset_q": offset_q,
                               "scale_drq": scale_drq,
                               "offset_pad": offset_pad,
                               "offset_rq": offset_rq,
                               "quantize_config": quantize_config,
                               "is_quantize": is_quantize,
                               "is_dequantize": is_dequantize,
                               "is_requantize": is_requantize,
                               "scale_sqrt": scale_sqrt,
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

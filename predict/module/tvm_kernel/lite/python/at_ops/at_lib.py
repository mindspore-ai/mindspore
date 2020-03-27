# Copyright 2019 Huawei Technologies Co., Ltd
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
"""
This module is rule to generation tvm operate, call by at_gen_strip.py
"""
import numpy as np
import tvm
import topi
from topi.image import resize
from topi.nn import mirror_pad
from topi import tag
import topi.testing

from arm_cpu.deconv import _conv_spatial_pack_deconv, schedule_conv2d_nchw_arm_cpu_deconv
from arm_cpu.conv2d import _conv_spatial_pack_asm, schedule_conv2d_nchw_arm_cpu
from arm_cpu.matmul import _matmul_spatial_pack_asm, _matmul_schedule_asm
from arm_cpu.depthwise_conv2d import _depthwise_spatial_pack, schedule_depthwise_conv2d_nchw_arm
from config_tool import activation_enum_map

map_conv = {
    'Convolution': "Conv2D",
    'ConvolutionDepthwise': "DepthwiseConv2D",
    'Deconvolution': "DeConv2D",
    'DeConvolutionDepthwise': "DeDepthwiseConv2D",
}


def Genlib(sch, tensor_list, device, opname, lib_path, print_lower=False):
    if print_lower:
        print(tvm.lower(sch, tensor_list, simple_mode=True))
    ctx = tvm.context(device, 0)
    func_o = tvm.build(sch, tensor_list, device + " --system-lib", name=opname)
    func_so = tvm.build(sch, tensor_list, device, name=opname)
    func_o.save(lib_path + opname + ".o", "o")
    return func_o, func_so, ctx


def AsType(as_input, dtype):
    if as_input.dtype == dtype:
        return as_input
    return tvm.compute(as_input.shape,
                       lambda *i: as_input(*i).astype(dtype),
                       tag="injective")


@tvm.tag_scope(tag=tag.ELEMWISE)
def TopiNNrelu6(x):
    return tvm.compute(x.shape, lambda *i: tvm.min(tvm.max(x(*i), tvm.const(0, x.dtype)), tvm.const(6, x.dtype)))


def TopiActivation(in_tensor, a_type, memcpy=False):
    '''
    activativation
    Args:
        in_tensor:
        a_type:
        memcpy:

    Returns:
    '''
    if a_type == 'NO_ACTIVATION':
        if memcpy:
            return tvm.compute(in_tensor.shape, lambda *i: in_tensor[i], tag=tag.ELEMWISE)
        return in_tensor
    if a_type == 'RELU':
        return topi.nn.relu(in_tensor)
    if a_type == 'RELU6':
        return TopiNNrelu6(in_tensor)
    if a_type == 'SIGMOID':
        if in_tensor.dtype in ["uint8", "int8", "uint32", "int32"]:
            a_fp32 = AsType(in_tensor, 'float32')
            out_tensor = topi.sigmoid(a_fp32)
            return AsType(out_tensor, in_tensor.dtype)
        return topi.sigmoid(in_tensor)
    raise ValueError("not support activation type" + a_type)


def Deconv(device="llvm", lib_path="./", optype=None,
           ndim=None, dtype=None, kernels=None,
           strides=None, pad=None, dilations=None,
           hasbias=None, activation_type=None,
           config_entity=None, impl_dtype=None,
           use_arm32=False, cfg=None):
    '''
    Deconvolution
    Args:
        device:
        lib_path:
        optype:
        ndim:
        dtype:
        kernels:
        strides:
        pad:
        dilations:
        hasbias:
        activationType:
        configEntity:
        impl_dtype:
        use_arm32:
        cfg:

    Returns:
    '''
    if cfg is None:
        cfg = {'CI': tvm.var('ci'), 'VH': 2, 'VW': 2, 'VC': 4, 'VI': 4,
               'tile_oh': 2, 'tile_ow': 2, 'tile_co': 4,
               'ann_reduce': ['none', 'none'],
               "ann_spatial": ['none', 'none', 'none']
               }
    has_bias = hasbias
    batch = tvm.var("batch")
    in_channel = tvm.var("in_channel")
    in_height, in_width = tvm.var("in_height"), tvm.var("in_width")
    kh, kw = kernels
    ow = cfg['VW']
    oh = cfg['VH']
    oc = cfg['VC']
    op_name = "%s_ndim%d_%s_k%d_s%d_p%d%d%d%d_d%d_act%d_vc%d_vh%d_vw%d_hasbias%d" % (\
               map_conv[optype], ndim, dtype,\
               kh, strides[0], pad[0], pad[1], pad[2], pad[3], dilations[0],\
               activation_enum_map[activation_type], oc, oh, ow, hasbias)
    opname = op_name
    print("DEconv", opname, config_entity)

    if impl_dtype is None:
        impl_dtype = dtype

    out_channel = tvm.var("out_channel")

    # define placeholder
    input_tensor = in_tensor = tvm.placeholder((batch, in_channel, in_height, in_width, 4), \
                                               dtype=dtype, name='in_tensor')
    temp_tensor = kernel_tensor = tvm.placeholder((in_channel*4, out_channel, kh, kw), dtype=dtype, \
                                                  name='kernel_tensor')
    if has_bias:
        bias = tvm.placeholder((out_channel,), dtype=dtype, name='bias')
        bias1 = topi.reshape(bias, (out_channel, 1, 1))

    if impl_dtype != dtype:
        input_tensor = AsType(input_tensor, impl_dtype)
        temp_tensor = AsType(temp_tensor, impl_dtype)
        if has_bias:
            bias1 = AsType(bias1, impl_dtype)

    # define compute & schedule
    cfg1 = (True, 1, 1, 1) if cfg is None else (True, cfg["tile_oh"], cfg["tile_ow"], cfg["tile_co"])
    out_tensor = _conv_spatial_pack_deconv(cfg1, input_tensor, temp_tensor, out_dtype=impl_dtype)

    if has_bias:
        out_tensor = tvm.compute(out_tensor.shape, lambda n, co, h, w, c4: \
            out_tensor[n, co, h, w, c4] + bias1[co*4 + c4][0][0], tag="injective")
    out_tensor = TopiActivation(out_tensor, activation_type)
    if impl_dtype != dtype:
        out_tensor = AsType(out_tensor, dtype)

    # create schedule
    if use_arm32:
        s = tvm.create_schedule(out_tensor.op)
    else:
        s = schedule_conv2d_nchw_arm_cpu_deconv(cfg, [out_tensor])

    attr = [batch, in_channel, in_height, in_width, out_channel, in_tensor, kernel_tensor]
    if has_bias: attr.append(bias)
    attr.append(out_tensor)
    tensor_list = attr

    Genlib(s, tensor_list, device, opname, lib_path)


def ConvVar(device="llvm", lib_path="./", optype=None,\
            ndim=None, layout=None, dtype=None, kernels=None,\
            strides=None, pad=None, dilations=None,\
            hasbias=None, activation_type=None,\
            config_entity=None, impl_dtype=None, channel_multiplier=None,\
            use_arm32=False, cfg=None):
    '''
    convolution
    Args:
        device:
        lib_path:
        optype:
        ndim:
        layout:
        dtype:
        kernels:
        strides:
        pad:
        dilations:
        hasbias:
        activationType:
        configEntity:
        impl_dtype:
        channel_multiplier:
        use_arm32:
        cfg:

    Returns:
    '''
    use_depthwise = optype == 'ConvolutionDepthwise'
    use_deconv = optype == 'Deconvolution'
    use_deconv_depthwise = optype == 'DeConvolutionDepthwise'
    has_bias = hasbias

    ow = 1 if cfg is None else cfg['VW']
    oh = 1 if cfg is None else cfg['VH']
    oc = 1 if cfg is None else cfg['VC']
    kh, kw = kernels
    op_name = "%s_ndim%d_%s_k%d_s%d_p%d%d%d%d_d%d_act%d_vc%d_vh%d_vw%d_hasbias%d" % ( \
              map_conv[optype], ndim, dtype, \
              kh, strides[0], pad[0], pad[1], pad[2], pad[3], dilations[0], \
              activation_enum_map[activation_type], oc, oh, ow, hasbias)
    batch = tvm.var("batch")
    in_channel = tvm.var("in_channel")
    in_height, in_width = tvm.var("in_height"), tvm.var("in_width")
    pad_up, pad_down, pad_left, pad_right = pad
    opname = op_name

    print("Conv", opname, config_entity)

    if impl_dtype is None:
        impl_dtype = dtype

    if use_depthwise:
        multiplier = channel_multiplier
        out_channel = in_channel * multiplier
    elif use_deconv_depthwise:
        multiplier = channel_multiplier
        out_channel = in_channel * multiplier
    else:
        out_channel = tvm.var("out_channel")

    # define placeholder
    input_tensor = in_tensor = tvm.placeholder((batch, in_channel, in_height, in_width), dtype=dtype, name='in_tensor')

    if use_depthwise:
        temp_tensor = kernel_tensor = tvm.placeholder((in_channel, multiplier, kh, kw), dtype=dtype,\
                                                      name='kernel_tensor')
    elif use_deconv:
        temp_tensor = kernel_tensor = tvm.placeholder((in_channel, out_channel, kh, kw), dtype=dtype,\
                                                      name='kernel_tensor')
    elif use_deconv_depthwise:
        temp_tensor = kernel_tensor = tvm.placeholder((in_channel, multiplier, kh, kw), dtype=dtype,\
                                                      name='kernel_tensor')
    else:
        temp_tensor = kernel_tensor = tvm.placeholder((out_channel, in_channel, kh, kw), dtype=dtype,\
                                                      name='kernel_tensor')
    if has_bias:
        bias = tvm.placeholder((out_channel,), dtype=dtype, name='bias')
        bias1 = topi.reshape(bias, (out_channel, 1, 1))

    if impl_dtype != dtype:
        input_tensor = AsType(input_tensor, impl_dtype)
        temp_tensor = AsType(temp_tensor, impl_dtype)
        if has_bias:
            bias1 = AsType(bias1, impl_dtype)

    # define compute & schedule
    if pad_up != pad_down or pad_left != pad_right:
        input_tensor = topi.nn.pad(input_tensor, [0, 0, pad_up, pad_left], [0, 0, pad_down, pad_right], name='data_pad')
        padding = 0, 0
    else:
        padding = pad_up, pad_left
    if use_depthwise:
        cfg1 = (True, 1, 1, 1) if cfg is None else (True, cfg["tile_oh"], cfg["tile_ow"], cfg["tile_co"])
        out_tensor = _depthwise_spatial_pack(cfg1, input_tensor, temp_tensor, strides, padding, dilations,\
                                             out_dtype=impl_dtype)
    elif use_deconv:

        def GetInput(input_tensor, temp_tensor, padding):
            _, out_c, filter_h, filter_w = temp_tensor.shape
            if out_c is None:
                print("temp_tensor.shape err")
            stride_h, stride_w = strides
            # dilate stage
            dilated_input = topi.nn.dilate(input_tensor, [1, 1, stride_h, stride_w],
                                           name='DilatedInput')
            # padding stage
            fpad_top, fpad_left, fpad_bottom, fpad_right = topi.nn.get_pad_tuple(padding, (
                filter_h, filter_w))
            bpad_top = filter_h - 1 - fpad_top
            bpad_bottom = filter_h - 1 - fpad_bottom
            bpad_left = filter_w - 1 - fpad_left
            bpad_right = filter_w - 1 - fpad_right
            padded_input = topi.nn.pad(dilated_input, \
                                      [0, 0, bpad_top, bpad_left], \
                                      [0, 0, bpad_bottom, bpad_right], \
                                      name='PaddedInput')
            return padded_input

        special_deconv = kh == 2 and kw == 2 and strides[0] == 2 and strides[1] == 2
        # special_deconv = False
        if special_deconv:
            out_tensor = OptimalOut(input_tensor, temp_tensor, in_channel)
        else:
            out_tensor = BaseImplementation(input_tensor, temp_tensor, GetInput, layout, padding)
    elif use_deconv_depthwise:
        def GetInput(input_tensor, temp_tensor, padding):
            _, out_c, filter_h, filter_w = temp_tensor.shape
            if out_c is None:
                print("temp_tensor.shape err")
            stride_h, stride_w = strides
            # dilate stage
            dilated_input = topi.nn.dilate(input_tensor, [1, 1, stride_h, stride_w],
                                           name='DilatedInput')
            # padding stage
            fpad_top, fpad_left, fpad_bottom, fpad_right = topi.nn.get_pad_tuple(padding, (
                filter_h, filter_w))
            bpad_top = filter_h - 1 - fpad_top
            bpad_bottom = filter_h - 1 - fpad_bottom
            bpad_left = filter_w - 1 - fpad_left
            bpad_right = filter_w - 1 - fpad_right
            padded_input = topi.nn.pad(dilated_input, \
                                      [0, 0, bpad_top, bpad_left], \
                                      [0, 0, bpad_bottom, bpad_right], \
                                      name='PaddedInput')
            return padded_input

        temp_tensor = topi.flip(temp_tensor, axis=-1)
        temp_tensor = topi.flip(temp_tensor, axis=-2)
        out_tensor = topi.nn.depthwise_conv2d_nchw(GetInput(input_tensor, temp_tensor, padding), temp_tensor, (1, 1), \
                                                   padding, (1, 1), out_dtype=input_tensor.dtype)
    else:
        cfg1 = (True, 1, 1, 1) if cfg is None else (True, cfg["tile_oh"], cfg["tile_ow"], cfg["tile_co"])
        out_tensor = _conv_spatial_pack_asm(cfg1, input_tensor, temp_tensor, strides, padding, dilations,\
                                            out_dtype=impl_dtype)

    if has_bias:
        out_tensor = tvm.compute(out_tensor.shape, lambda n, co, h, w: out_tensor[n, co, h, w] + bias1[co][0][0],\
                                 tag="injective")
    out_tensor = TopiActivation(out_tensor, activation_type)
    if impl_dtype != dtype:
        out_tensor = AsType(out_tensor, dtype)

    # create schedule
    if use_arm32:
        s = tvm.create_schedule(out_tensor.op)
    elif use_depthwise:
        s = schedule_depthwise_conv2d_nchw_arm(cfg, [out_tensor])
    elif use_deconv:
        if special_deconv:
            s = tvm.create_schedule([out_tensor.op])
        else:
            s = topi.generic.schedule_conv2d_nchw([out_tensor])
    elif use_deconv_depthwise:
        s = tvm.create_schedule([out_tensor.op])
    else:
        s = schedule_conv2d_nchw_arm_cpu([out_tensor])

    # generate lib
    attr = [batch, in_channel, in_height, in_width, out_channel, in_tensor, kernel_tensor]
    tensor_list = [*attr, bias, out_tensor] if has_bias else [*attr, out_tensor]
    Genlib(s, tensor_list, device, opname, lib_path)


def BaseImplementation(input_tensor, temp_tensor, get_input, layout, padding):
    temp_tensor = topi.flip(temp_tensor, axis=-1)
    temp_tensor = topi.flip(temp_tensor, axis=-2)
    temp_tensor = topi.transpose(temp_tensor, axes=(1, 0, 2, 3))
    out_tensor = topi.nn.conv2d(get_input(input_tensor, temp_tensor, padding), temp_tensor, (1, 1), padding, (1, 1),
                                layout=layout, out_dtype=input_tensor.dtype)
    return out_tensor


def OptimalOut(input_tensor, temp_tensor, in_channel):
    '''
    deconv compute
    Args:
        input_tensor:
        temp_tensor:
        in_channel:

    Returns:
    '''
    temp_tensor = topi.transpose(temp_tensor, axes=(1, 0, 2, 3))
    out_shape = []
    for i in range(len(input_tensor.shape)):
        if i == 0:
            out_shape.append(input_tensor.shape[i])
            continue
        if i == 1:
            out_shape.append(temp_tensor.shape[0])
            continue
        out_shape.append(2 * input_tensor.shape[i])
    rc = tvm.reduce_axis((0, in_channel), name='rc')
    return tvm.compute(out_shape, lambda i, j, k, l:\
        tvm.sum(input_tensor[i, rc, k // 2, l // 2].astype(input_tensor.dtype) *\
                temp_tensor[j, rc, k % 2, l % 2].astype(input_tensor.dtype), axis=[rc]))


def Concat(device="llvm", lib_path="./",
           ndim=None, dtype=None, input_num=None, axis=None):
    '''
    concat
    Args:
        device:
        lib_path:
        all_tensors:
        ndim:
        dtype:
        input_num:
        axis:

    Returns:
    '''
    if axis >= ndim:
        return
    shapes = []
    for i in range(input_num):
        shape = []
        for j in range(ndim):
            if j == axis:
                shape.append(tvm.var("axis" + str(i)))
            else:
                shape.append(tvm.var("n" + str(j)))
        shapes.append(shape)
    in_tensor = [tvm.placeholder(shape, dtype=dtype, name='in_tensor%d' % i) for i, shape in enumerate(shapes)]
    opname = "Concat_ndim%d_%s_input_num%d_axis%d" % (ndim, dtype, input_num, axis)
    print(opname)

    # define compute
    out_tensor = topi.concatenate(tuple(in_tensor), axis)
    tensor_list = in_tensor + [out_tensor]
    if ndim < 5:
        s = topi.generic.schedule_concatenate(out_tensor)
    else:
        s = tvm.create_schedule(out_tensor.op)
    Genlib(s, tensor_list, device, opname, lib_path)


def Activation(device="llvm", lib_path="./",
               ndim=None, dtype=None, optype=None):
    '''
    activation
    Args:
        device:
        lib_path:
        ndim:
        dtype:
        optype:

    Returns:
    '''
    shape = [tvm.var("n" + str(i)) for i in range(ndim)]
    opname = "Activation_ndim%d_%s_%s" % (ndim, dtype, optype)
    print(opname)

    # define compute
    in_tensor = tvm.placeholder(shape, dtype=dtype, name='in_tensor')
    out_tensor = TopiActivation(in_tensor, optype, memcpy=True)
    tensor_list = [in_tensor, out_tensor]
    s = tvm.create_schedule(out_tensor.op)
    Genlib(s, tensor_list, device, opname, lib_path)


def BatchNorm(device="llvm", lib_path="./",
              ndim=None, dtype=None, optype=False, axis=None):
    '''
    batchnorm
    Args:
        device:
        lib_path:
        ndim:
        dtype:
        optype:
        axis:

    Returns:
    '''
    if axis >= ndim:
        return
    shape = [tvm.var("n" + str(i)) for i in range(ndim)]
    channel = shape[axis]
    eps = tvm.var("epsilon", dtype="float32")
    opname = optype + ("_ndim%d_%s_axis%d" % (ndim, dtype, axis))
    print(opname)

    # define compute
    in_tensor = tvm.placeholder(shape, dtype=dtype, name='in_tensor')
    mean = tvm.placeholder((channel,), dtype=dtype, name='mean')
    variance = tvm.placeholder((channel,), dtype=dtype, name='var')
    scale = tvm.placeholder((channel,), dtype=dtype, name='scale')
    offset = tvm.placeholder((channel,), dtype=dtype, name='offset')

    variance_sqrt = tvm.compute((channel,), lambda i: tvm.sqrt(variance[i] + eps.astype(dtype)))
    if optype == "TFBatchNorm":
        out_tensor = tvm.compute(shape, lambda *idx: ((in_tensor[idx] - mean[idx[axis]]) / variance_sqrt[idx[axis]]) *\
                                                     scale[idx[axis]] + offset[idx[axis]])
        tensor_list = [eps, in_tensor, scale, offset, mean, variance, out_tensor]
    elif optype == "CaffeBatchNorm":
        out_tensor = tvm.compute(shape, lambda *idx: (in_tensor[idx] - mean[idx[axis]]) / variance_sqrt[idx[axis]])
        tensor_list = [eps, in_tensor, mean, variance, out_tensor]
    elif optype == "CaffeScale":
        out_tensor = tvm.compute(shape, lambda *idx: in_tensor[idx] * scale[idx[axis]] + offset[idx[axis]])
        tensor_list = [in_tensor, scale, offset, out_tensor]
    elif optype == "TFBiasAdd":
        out_tensor = tvm.compute(shape, lambda *idx: in_tensor[idx] + offset[idx[axis]])
        tensor_list = [in_tensor, offset, out_tensor]
    else:
        raise RuntimeError("no support for {}".format(optype))

    # define schedule & generate lib
    s = tvm.create_schedule(out_tensor.op)
    Genlib(s, tensor_list, device, opname, lib_path)


def Pooling(device="llvm", lib_path="./",
            ndim=None, dtype=None, pooling_mode=None, kernel=None, stride=None, pad=None, caffe_mode=None,
            use_global=False):
    '''
    pooling
    Args:
        device:
        lib_path:
        ndim:
        dtype:
        pooling_mode:
        kernel:
        stride:
        pad:
        caffe_mode:
        use_global:

    Returns:
    '''
    shape = [tvm.var("n" + str(i)) for i in range(0, ndim)]
    layout = 'NCHW'
    if use_global:
        opname = "GlobalPooling_ndim%d_%s_%s" % (ndim, dtype, pooling_mode)
    else:
        kernel_h, kernel_w = kernel
        stride_h, stride_w = stride
        pad_up, pad_down, pad_left, pad_right = pad
        if pad_up == 0 and pad_down == 0 and pad_left == 0 and pad_right == 0 and caffe_mode:
            caffe_mode = False
        opname = "Pooling_ndim%d_%s_%s_kernel%d%d_stride%d%d_pad%d%d%d%d%s" \
                 % (ndim, dtype, pooling_mode, kernel_h, kernel_w, stride_h, stride_w,
                    pad_up, pad_down, pad_left, pad_right, "_caffe" if caffe_mode else "")
    print(opname)

    # define compute
    in_tensor = tvm.placeholder(shape, dtype=dtype, name='in_tensor')
    if use_global:
        out_tensor = topi.nn.global_pool(in_tensor, pool_type=pooling_mode, layout=layout)
        sch = topi.generic.schedule_adaptive_pool(out_tensor)
    else:
        out_tensor = topi.nn.pool(in_tensor,
                                  kernel=(kernel_h, kernel_w),
                                  stride=(stride_h, stride_w),
                                  padding=(pad_up, pad_left, pad_down, pad_right),
                                  pool_type=pooling_mode,
                                  ceil_mode=False,
                                  layout=layout,
                                  count_include_pad=False)
        sch = topi.generic.schedule_pool(out_tensor, layout)
    tensor_list = [in_tensor, out_tensor]
    Genlib(sch, tensor_list, device, opname, lib_path, print_lower=False)


def Eltwise(device="llvm", lib_path="./",
            ndim_a=None, ndim_b=None, dtype=None, mode=None):
    '''
    eltwise
    Args:
        device:
        lib_path:
        ndim_a:
        ndim_b:
        dtype:
        mode:

    Returns:
    '''
    ndim_max = max(ndim_a, ndim_b)
    shape = [tvm.var("n" + str(i)) for i in range(ndim_max)]
    shape_b1 = [dim if i == 1 else 1 for i, dim in enumerate(shape)]
    shape_a = shape[ndim_max - ndim_a:] if ndim_a else (1,)
    shape_b = shape[ndim_max - ndim_b:] if ndim_b == ndim_a else shape_b1 if ndim_b == 1 else (1,)
    opname = "Eltwise_%s_ndimA%d_ndimB%d_%s" % (mode, ndim_a, ndim_b, dtype)
    print(opname)

    # define compute
    in_tensor = tvm.placeholder(shape_a, dtype=dtype, name='in_tensor')
    b_tensor = tvm.placeholder(shape_b, dtype=dtype, name='b_tensor')

    topi_funs = {
        'add': topi.add,
        'subtract': topi.subtract,
        'multiply': topi.multiply,
        'divide': topi.divide,
        'maximum': topi.maximum,
        'minimum': topi.minimum,
    }

    out_tensor = topi_funs[mode](in_tensor, b_tensor)
    tensor_list = [in_tensor, b_tensor, out_tensor]
    s = topi.generic.schedule_elemwise(out_tensor)
    Genlib(s, tensor_list, device, opname, lib_path)


def Softmax(device="llvm", lib_path="./",
            ndim=None, dtype=None, axis=None):
    '''
    softmax
    Args:
        device:
        lib_path:
        ndim:
        dtype:
        axis:

    Returns:
    '''
    if axis >= ndim:
        return
    shape = [tvm.var("n" + str(i)) for i in range(ndim)]
    opname = "Softmax_ndim%d_%s_axis%s" % (ndim, dtype, axis)
    print(opname)

    # define compute
    in_tensor = tvm.placeholder(shape, dtype=dtype, name='in_tensor')
    out_tensor = topi.nn.softmax(in_tensor, axis)
    tensor_list = [in_tensor, out_tensor]
    s = topi.generic.schedule_elemwise(out_tensor)
    Genlib(s, tensor_list, device, opname, lib_path)


def Resize(device="llvm", lib_path="./",
           ndim=None, dtype=None, method=None, align_corners=None):
    '''
    resize
    Args:
        device:
        lib_path:
        ndim:
        dtype:
        method:
        align_corners:

    Returns:
    '''
    shape = [tvm.var("n" + str(i)) for i in range(ndim)]
    new_height = tvm.var("newHeight")
    new_width = tvm.var("new_width")
    opname = "Resize_ndim%d_%s_%s_%s" % (ndim, dtype, method, "Align" if align_corners else "NotAlign")
    print(opname)

    # define compute
    in_tensor = tvm.placeholder(shape, dtype=dtype, name='in_tensor')
    out_tensor = resize(in_tensor, [new_height, new_width], align_corners=align_corners, method=method)
    tensor_list = [new_height, new_width, in_tensor, out_tensor]
    s = topi.generic.schedule_injective(out_tensor)
    Genlib(s, tensor_list, device, opname, lib_path)


def Mean(device="llvm", lib_path="./",
         ndim=None, dtype=None, axis=None, keep_dims=None):
    '''
    mean
    Args:
        device:
        lib_path:
        ndim:
        dtype:
        axis:
        keepDims:

    Returns:
    '''
    if axis[-1] >= ndim:
        return
    shape = [tvm.var("n" + str(i)) for i in range(ndim)]
    axis_str = ""
    for dim in axis:
        axis_str += str(dim)
    opname = "Mean_ndim%d_%s_axis%s_%s" % (ndim, dtype, axis_str, "keepDims" if keep_dims else "notkeepDims")
    print(opname)

    # define compute
    in_tensor = tvm.placeholder(shape, dtype=dtype, name='in_tensor')
    c_shape = shape[:]
    reduced_num = 1
    for dim in axis:
        c_shape[dim] = 1
        reduced_num *= shape[dim]

    def _ComputeSum(*b_idx):
        reduce_axis = [tvm.reduce_axis((0, shape[dim])) for dim in axis]
        a_idx = list(b_idx)
        for i, dim in enumerate(axis):
            a_idx[dim] = reduce_axis[i]
        a_idx = tuple(a_idx)
        return tvm.sum(in_tensor[a_idx], axis=reduce_axis)

    out_tensor = tvm.compute(c_shape, _ComputeSum)
    out_tensor = tvm.compute(c_shape, lambda *i: out_tensor(*i) / reduced_num)
    if not keep_dims:
        out_tensor = topi.squeeze(out_tensor, axis)

    # define schedule & generate lib
    tensor_list = [in_tensor, out_tensor]
    s = tvm.create_schedule(out_tensor.op)
    Genlib(s, tensor_list, device, opname, lib_path)


def CaffeCrop(device="llvm", lib_path="./",
              ndim=None, dtype=None, axis=None):
    '''
    caffe crop op
    Args:
        device:
        lib_path:
        ndim:
        dtype:
        axis:

    Returns:
    '''
    shape = [tvm.var("n" + str(i)) for i in range(axis)]
    shape_a = shape[:]
    shape_b = shape[:]
    offsets = []
    for i in range(axis, ndim):
        shape_a.append(tvm.var("nA" + str(i)))
        shape_b.append(tvm.var("nB" + str(i)))
        offsets.append(tvm.var("offset" + str(i)))
    opname = "CaffeCrop_ndim%d_%s_axis%d" % (ndim, dtype, axis)
    print(opname)

    # define compute
    in_tensor = tvm.placeholder(shape_a, dtype=dtype, name='in_tensor')
    b_tensor = tvm.placeholder(shape_b, dtype=dtype, name='b_tensor')
    begin = [0] * axis + offsets
    end = shape_a[:]
    for i in range(axis, len(shape_a)):
        end[i] = offsets[i - axis] + shape_b[i]
    shape_c = [end[i] - begin[i] for i in range(ndim)]

    def _Compute(*C_idx):
        a_idx = [idx + begin[i] for i, idx in enumerate(list(C_idx))]
        a_idx = tuple(a_idx)
        return in_tensor[a_idx]

    out_tensor = tvm.compute(shape_c, _Compute)
    tensor_list = offsets + [in_tensor, b_tensor, out_tensor]

    s = tvm.create_schedule(out_tensor.op)
    Genlib(s, tensor_list, device, opname, lib_path)


def FullConnection(device="llvm", lib_path="./",
                   ndim_a=None, dtype=None, has_bias=None):
    '''
    full connection
    Args:
        device:
        lib_path:
        ndim_a:
        dtype:
        hasBias:

    Returns:
    '''
    n_dim, ci, h_dim, kernel_tensor = (tvm.var("n_dim"), tvm.var("out_tensor"), tvm.var("h_dim"), \
                                       tvm.var("kernel_tensor"))
    co = tvm.var("co")
    if ndim_a == 4:
        shape_a = (n_dim, ci, h_dim, kernel_tensor)
        chw = ci * h_dim * kernel_tensor
    else:
        shape_a = (n_dim, ci)
        chw = ci
    shape_w = (co, chw)
    opname = "FullConnection_ndimA%d_%s_%s" % (ndim_a, dtype, "hasBias" if has_bias else "notHasBias")
    is_var = True
    vh, vw, vc = 1, 1, 1
    print(opname)

    in_tensor = tvm.placeholder(shape_a, dtype=dtype, name='in_tensor')
    kernel_tensor = tvm.placeholder(shape_w, dtype=dtype, name='kernel_tensor')
    input_tensor = topi.reshape(in_tensor, (n_dim, chw)) if len(shape_a) == 4 else in_tensor

    out_tensor = _matmul_spatial_pack_asm((is_var, 0, ci, vh, vw, vc), input_tensor, kernel_tensor, \
                                          layout='NC', out_dtype=dtype)
    if has_bias:
        bias = tvm.placeholder((co,), dtype=dtype, name='bias')
        out_tensor = tvm.compute((n_dim, co), lambda n, co: out_tensor[n, co] + bias[co], tag='injective')

    tensor_list = [in_tensor, kernel_tensor, bias, out_tensor] if has_bias else [in_tensor, kernel_tensor, out_tensor]
    cfg = {'is_var': is_var, 'is_transpose': 0, 'core_id': 0, 'CI': ci, 'VH': vh, 'VW': vw, 'VC': vc}
    s = _matmul_schedule_asm(cfg, [out_tensor])
    Genlib(s, tensor_list, device, opname, lib_path)


def Power(device="llvm", lib_path="./",
          ndim=None, dtype=None):
    '''
    power
    Args:
        device:
        lib_path:
        ndim:
        dtype:

    Returns:
    '''
    shape = [tvm.var("n" + str(i)) for i in range(ndim)]
    power = tvm.var("power", dtype="float32")
    scale = tvm.var("scale", dtype="float32")
    shift = tvm.var("shift", dtype="float32")
    opname = "Power_ndim%d_%s" % (ndim, dtype)
    print(opname)

    in_tensor = tvm.placeholder(shape, dtype=dtype, name='in_tensor')
    out_tensor = tvm.compute(shape, lambda *i: tvm.power(in_tensor[i] * scale.astype(in_tensor.dtype) + \
                                                         shift.astype(in_tensor.dtype), \
                                                         power.astype(in_tensor.dtype)))
    tensor_list = [power, scale, shift, in_tensor, out_tensor]

    s = tvm.create_schedule(out_tensor.op)
    Genlib(s, tensor_list, device, opname, lib_path)


def CaffePReLU(device="llvm", lib_path="./",
               ndim=None, dtype=None, channel_shared=None):
    '''
    caffe prelu
    Args:
        device:
        lib_path:
        ndim:
        dtype:
        channel_shared:

    Returns:
    '''
    shape = [tvm.var("n" + str(i)) for i in range(ndim)]
    channel = 1 if channel_shared else shape[1]
    opname = "CaffePReLU_ndim%d_%s_%s" % (ndim, dtype,
                                          "channelShared" if channel_shared else "channelNotShared")
    print(opname)

    in_tensor = tvm.placeholder(shape, dtype=dtype, name='in_tensor')
    slope = tvm.placeholder((channel,), dtype=dtype, name='slope')
    if channel_shared:
        out_tensor = tvm.compute(shape, lambda *idx: tvm.if_then_else(in_tensor[idx] >= 0, in_tensor[idx],\
                                                                      in_tensor[idx] * slope[0]))
    else:
        out_tensor = tvm.compute(shape, lambda *idx: tvm.if_then_else(in_tensor[idx] >= 0, in_tensor[idx],\
                                                                      in_tensor[idx] * slope[idx[1]]))

    tensor_list = [in_tensor, slope, out_tensor]
    s = tvm.create_schedule(out_tensor.op)
    Genlib(s, tensor_list, device, opname, lib_path)


def Pad(device="llvm", lib_path="./",
        ndim=None, dtype=None, paddingmode=None):
    '''
    pad
    Args:
        device:
        lib_path:
        ndim:
        dtype:
        paddingmode:

    Returns:
    '''
    shape = [tvm.var("n" + str(i)) for i in range(ndim)]
    pad_before = [tvm.var("pad_before" + str(i)) for i in range(ndim)]
    pad_after = [tvm.var("pad_after" + str(i)) for i in range(ndim)]
    pad_before_const = [0, 0] + pad_before[2:]
    pad_after_const = [0, 0] + pad_after[2:]
    paddings = [None] * 2 * len(shape)
    paddings[0:: 2] = pad_before
    paddings[1:: 2] = pad_after
    pad_value = 0
    opname = "Pad_ndim%d_%s_%s" % (ndim, dtype, paddingmode)
    print(opname)

    # define compute
    in_tensor = tvm.placeholder(shape, dtype=dtype, name='in_tensor')
    if paddingmode == "CONSTANT":
        out_tensor = topi.nn.pad(in_tensor, pad_before_const, pad_after_const, pad_value=pad_value, name='out_tensor')
    else:
        out_tensor = mirror_pad(in_tensor, pad_before_const, pad_after_const, mode=paddingmode, name='out_tensor')
    tensor_list = paddings + [in_tensor, out_tensor]
    def SchedulePad(inputs):
        s = tvm.create_schedule(inputs.op)
        if s[inputs].op.axis:
            s[inputs].parallel(s[inputs].op.axis[1])
        return s

    s = SchedulePad(out_tensor)
    Genlib(s, tensor_list, device, opname, lib_path)


def MatMul(device="llvm", lib_path="./",
           ndim_a=None, ndim_b=None, dtype=None, transpose_a=None, transpose_b=None):
    '''
    matmul
    Args:
        device:
        lib_path:
        ndim_a:
        ndim_b:
        dtype:
        transpose_a:
        transpose_b:

    Returns:
    '''
    m, k, n_dim = tvm.var("m"), tvm.var("k"), tvm.var("n_dim")
    a_shape = (m, k) if not transpose_a else (k, m)
    b_shape = (k, n_dim) if not transpose_b else (n_dim, k)
    opname = "MatMul_ndimA%d_ndimB%d_%s_%d_%d" % (ndim_a, ndim_b, dtype, transpose_a, transpose_b)
    print(opname)

    # define compute
    in_tensor = tvm.placeholder(a_shape, dtype=dtype, name='in_tensor')
    b_tensor = tvm.placeholder(b_shape, dtype=dtype, name='b_tensor')
    out_tensor = topi.matmul(in_tensor, b_tensor, transpose_a, transpose_b)
    tensor_list = [in_tensor, b_tensor, out_tensor]
    s = topi.generic.schedule_elemwise(out_tensor)
    Genlib(s, tensor_list, device, opname, lib_path)


def Stack(device="llvm", lib_path="./",
          ndim=None, dtype=None, input_num=None, axis=None):
    '''
    stack
    Args:
        device:
        lib_path:
        ndim:
        dtype:
        input_num:
        axis:

    Returns:
    '''
    if axis > ndim:
        return
    shape = [tvm.var("n" + str(i)) for i in range(ndim)]
    shapes = [shape] * input_num
    in_tensor = [tvm.placeholder(shape, dtype=dtype, name='in_tensor%d' % i) for i, shape in enumerate(shapes)]
    opname = "Stack_ndim%d_%s_input_num%d_axis%d" % (ndim, dtype, input_num, axis)
    print(opname)

    input_tensor = [topi.expand_dims(ai, axis) for ai in in_tensor]
    out_tensor = topi.concatenate(tuple(input_tensor), axis=axis)
    tensor_list = in_tensor + [out_tensor]
    if ndim < 4:
        s = topi.generic.schedule_concatenate(out_tensor)
    else:
        s = tvm.create_schedule(out_tensor.op)
    Genlib(s, tensor_list, device, opname, lib_path)


def ArgMax(device="llvm", lib_path="./",
           ndim=None, dtype=None, axis=None, keep_dims=None, top_k=None,
           out_dtype=None):
    '''
    argmax
    Args:
        device:
        lib_path:
        ndim:
        dtype:
        axis:
        keepDims:
        top_k:
        out_dtype:

    Returns:
    '''
    if axis >= ndim:
        return
    shape = [tvm.var("n" + str(i)) for i in range(ndim)]
    opname = "ArgMax_ndim%d_%s_axis%d_%s_top%d_%s" \
             % (ndim, dtype, axis, "keepDims" if keep_dims else "notKeepDims", top_k, out_dtype)
    print(opname)

    in_tensor = tvm.placeholder(shape, dtype=dtype, name='in_tensor')
    out_tensor = topi.argmax(in_tensor, axis=axis, keepdims=keep_dims)
    out_tensor = AsType(out_tensor, out_dtype)
    tensor_list = [in_tensor, out_tensor]
    s = tvm.create_schedule(out_tensor.op)
    Genlib(s, tensor_list, device, opname, lib_path)


def Exp(device="llvm", lib_path="./",
        ndim=None, dtype=None):
    '''
    exp
    Args:
        device:
        lib_path:
        ndim:
        dtype:

    Returns:
    '''
    shape = [tvm.var("n" + str(i)) for i in range(ndim)]
    opname = "Exp_ndim%d_%s" % (ndim, dtype)
    print(opname)

    # define compute
    in_tensor = tvm.placeholder(shape, dtype=dtype, name='in_tensor')
    if 'int' in dtype:
        input_tensor = AsType(in_tensor, 'float32')
        out_tensor = topi.exp(input_tensor)
        out_tensor = AsType(out_tensor, in_tensor.dtype)
    else:
        out_tensor = topi.exp(in_tensor)
    tensor_list = [in_tensor, out_tensor]
    s = topi.generic.schedule_injective(out_tensor)
    Genlib(s, tensor_list, device, opname, lib_path)


def Cast(device="llvm", lib_path="./",
         ndim=None, src_dtype=None, dst_dtype=None):
    '''
    cast
    Args:
        device:
        lib_path:
        ndim:
        src_dtype:
        dst_dtype:

    Returns:
    '''
    shape = [tvm.var("n" + str(i)) for i in range(ndim)]
    opname = "Cast_ndim%d_%s_%s" % (ndim, src_dtype, dst_dtype)
    print(opname)

    # define compute
    in_tensor = tvm.placeholder(shape, dtype=src_dtype, name='in_tensor')
    out_tensor = topi.cast(in_tensor, dst_dtype)
    tensor_list = [in_tensor, out_tensor]
    s = topi.generic.schedule_injective(out_tensor)
    Genlib(s, tensor_list, device, opname, lib_path)


def ExpandDims(device="llvm", lib_path="./",
               ndim=None, axis=None, dtype=None):
    '''
    expand dims
    Args:
        device:
        lib_path:
        ndim:
        axis:
        dtype:

    Returns:
    '''
    if axis > ndim:
        return
    shape = [tvm.var("n" + str(i)) for i in range(ndim)]
    opname = "ExpandDim_ndim%d_%s_axis%d" % (ndim, dtype, axis)
    print(opname)

    # define compute
    in_tensor = tvm.placeholder(shape, dtype=dtype, name='in_tensor')
    out_tensor = topi.expand_dims(in_tensor, axis=axis)
    tensor_list = [in_tensor, out_tensor]
    s = topi.generic.schedule_injective(out_tensor)
    Genlib(s, tensor_list, device, opname, lib_path)


def Tile(device="llvm", lib_path="./",
         ndim=None, dtype=None):
    '''
    tile
    Args:
        device:
        lib_path:
        ndim:
        dtype:

    Returns:
    '''
    shape = [tvm.var("n" + str(i)) for i in range(ndim)]
    multiples = [tvm.var("k" + str(i)) for i in range(ndim)]
    opname = "Tile_ndim%d_%s" % (ndim, dtype)
    print(opname)

    def _Compute(*C_idx):
        a_idx = [tvm.floordiv(idx, multiples[i]) for i, idx in enumerate(list(C_idx))]
        a_idx = tuple(a_idx)
        return in_tensor[a_idx]

    # define compute
    in_tensor = tvm.placeholder(shape, dtype=dtype, name='in_tensor')  # tvm 0.6-dev: topi.tile
    shape_c = (np.array(shape) * np.array(multiples)).tolist()
    out_tensor = tvm.compute(shape_c, _Compute)

    tensor_list = multiples + [in_tensor, out_tensor]
    s = topi.generic.schedule_injective(out_tensor)
    Genlib(s, tensor_list, device, opname, lib_path)


def Range(device="llvm", lib_path="./",
          out_dtype=None):
    '''
    range
    Args:
        device:
        lib_path:
        out_dtype:

    Returns:
    '''
    start = tvm.var("start")
    delta = tvm.var("delta")
    opname = "Range_ndim_" + out_dtype
    print(opname)

    out_tensor = tvm.compute((tvm.var("n0"),), lambda i: start.astype(out_dtype) + delta.astype(out_dtype) * i, \
                             name='out_tensor')
    out_tensor = AsType(out_tensor, out_dtype)
    tensor_list = [start, delta, out_tensor]
    s = topi.generic.schedule_injective(out_tensor)
    Genlib(s, tensor_list, device, opname, lib_path)


def Split(device="llvm", lib_path="./",
          ndim=None, dtype=None, output_num=None, axis=None):
    '''
    split
    Args:
        device:
        lib_path:
        ndim:
        dtype:
        output_num:
        axis:

    Returns:
    '''
    if axis >= ndim:
        return
    size_splits = [tvm.var("split" + str(i)) for i in range(output_num)]
    a_shape = [tvm.var("n" + str(i)) for i in range(axis)] \
               + [np.sum(size_splits)] \
               + [tvm.var("n" + str(i)) for i in range(axis + 1, ndim)]
    c_shapes = []
    for i in range(output_num):
        c_shape = []
        for j in range(ndim):
            if j == axis:
                c_shape.append(tvm.var("split" + str(i)))
            else:
                c_shape.append(tvm.var("n" + str(j)))
        c_shapes.append(c_shape)
    indices_or_sections = np.cumsum(size_splits).tolist()[:-1]
    opname = "Split_ndim%d_%s_output_num%d_axis%d" % (ndim, dtype, output_num, axis)
    print(opname)

    # define compute
    in_tensor = tvm.placeholder(a_shape, dtype=dtype, name='in_tensor')

    def _Compute(*C_idx):
        a_idx = list(C_idx)
        a_idx[axis] += idx_shift
        a_idx = tuple(a_idx)
        return in_tensor[a_idx]

    indices_or_sections_add0 = [0] + indices_or_sections
    out_tensor = []
    for i in range(output_num):
        idx_shift = indices_or_sections_add0[i]
        ci = tvm.compute(c_shapes[i], _Compute)
        out_tensor.append(ci)
    tensor_list = size_splits + [in_tensor] + out_tensor

    s = topi.generic.schedule_injective(out_tensor)
    Genlib(s, tensor_list, device, opname, lib_path)

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

"""convolution vmap impl"""
from __future__ import absolute_import

import mindspore.numpy as mnp
from mindspore.ops import constexpr
from mindspore.ops.primitive import _primexpr
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.operations import nn_ops as nps
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops.primitive import Primitive
from mindspore.ops._vmap.vmap_base import vmap_rules_getters, vmap_general_preprocess, \
    _raise_value_error, _vmap_update_prim_attr, _vmap_clone_prim


@vmap_rules_getters.register(P.Conv2D)
@vmap_rules_getters.register(P.Conv3D)
def get_conv_vmap_rule(prim, axis_size):
    """Vmap rule for `Conv2D` and `Conv3D` operations."""
    if isinstance(prim, str):
        prim = Primitive(prim)

    attr_list = [prim.name, prim.group, prim.data_format]
    new_prim = _vmap_clone_prim(prim)

    def vmap_rule(input_bdim, weight_bdim):
        is_all_none, result = vmap_general_preprocess(prim, input_bdim, weight_bdim)
        if is_all_none:
            return result
        return _conv_vmap_rule(new_prim, axis_size, input_bdim, weight_bdim, attr_list)

    return vmap_rule


@vmap_rules_getters.register(P.Conv2DTranspose)
@vmap_rules_getters.register(P.Conv2DBackpropInput)
def get_conv2d_transpose_vmap_rule(prim, axis_size):
    """Vmap rule for `Conv2DTranspose` and `Conv2DBackpropInput` operations."""
    if isinstance(prim, str):
        prim = Primitive(prim)

    attr_list = [prim.name, prim.group, prim.data_format]
    new_prim = _vmap_clone_prim(prim)

    def vmap_rule(dout_bdim, weight_bdim, input_size_bdim):
        is_all_none, result = vmap_general_preprocess(prim, dout_bdim, weight_bdim, input_size_bdim)
        if is_all_none:
            return result
        return _conv_transpose_vmap_rule(new_prim, axis_size, dout_bdim, \
                                         weight_bdim, input_size_bdim, attr_list)

    return vmap_rule


@vmap_rules_getters.register(P.Conv3DTranspose)
def get_conv3d_transpose_vmap_rule(prim, axis_size):
    """Vmap rule for `Conv3DTranspose` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)

    attr_list = [prim.name, prim.group, prim.data_format]
    new_prim = _vmap_clone_prim(prim)

    def vmap_rule(dout_bdim, weight_bdim):
        is_all_none, result = vmap_general_preprocess(prim, dout_bdim, weight_bdim)
        if is_all_none:
            return result
        return _conv_transpose_vmap_rule(new_prim, axis_size, dout_bdim, weight_bdim, None, attr_list)

    return vmap_rule


@vmap_rules_getters.register(nps.Conv3DBackpropInput)
def get_conv3d_backprop_input_vmap_rule(prim, axis_size):
    """Vmap rule for `Conv3DBackpropInput` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)

    attr_list = [prim.name, prim.group, prim.data_format]
    new_prim = _vmap_clone_prim(prim)

    def vmap_rule(weight_bdim, dout_bdim, input_size_bdim):
        is_all_none, result = vmap_general_preprocess(prim, weight_bdim, dout_bdim, input_size_bdim)
        if is_all_none:
            return result
        return _conv_transpose_vmap_rule(new_prim, axis_size, dout_bdim, \
                                         weight_bdim, input_size_bdim, attr_list)

    return vmap_rule


@vmap_rules_getters.register(G.Conv2DBackpropFilter)
def get_conv2d_backprop_filter_vmap_rule(prim, axis_size):
    """Vmap rule for `Conv2DBackpropFilter` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)

    attr_list = [prim.name, prim.group, prim.data_format]
    new_prim = _vmap_clone_prim(prim)

    def vmap_rule(dout_bdim, input_x_bdim, weight_size_bdim):
        is_all_none, result = vmap_general_preprocess(prim, dout_bdim, input_x_bdim, weight_size_bdim)
        if is_all_none:
            return result
        return _conv_backprop_filter_vmap_rule(new_prim, axis_size, dout_bdim, \
                                               input_x_bdim, weight_size_bdim, attr_list)

    return vmap_rule


@vmap_rules_getters.register(G.Conv3DBackpropFilter)
def get_conv3d_backprop_filter_vmap_rule(prim, axis_size):
    """Vmap rule for `Conv3DBackpropFilter` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)

    attr_list = [prim.name, prim.group, prim.data_format]
    new_prim = _vmap_clone_prim(prim)

    def vmap_rule(input_x_bdim, dout_bdim, weight_size_bdim):
        is_all_none, result = vmap_general_preprocess(prim, input_x_bdim, dout_bdim, weight_size_bdim)
        if is_all_none:
            return result
        return _conv_backprop_filter_vmap_rule(new_prim, axis_size, dout_bdim, \
                                               input_x_bdim, weight_size_bdim, attr_list)

    return vmap_rule


@_primexpr
def _get_reshape_src_dim(data_dim, cmp_dim):
    """Get source dim for reshape"""
    if data_dim > cmp_dim:
        expand_dim = cmp_dim
        merge_dim = data_dim + 1
    else:
        expand_dim = cmp_dim + 1
        merge_dim = data_dim
    return expand_dim, merge_dim


@_primexpr
def _get_merge_shape(src_dim, dst_dim, shape):
    """Get new shape for merging the src_dim and dst_dim. The dst_dim is the value after removing src_dim."""
    new_shape = [shape[i] for i in range(len(shape)) if i != src_dim]
    new_shape[dst_dim] *= shape[src_dim]
    return tuple(new_shape)


def _reshape_merge_dims(src_dim, dst_dim, target):
    """Reshape target by merging the src_dim and dst_dim."""
    shape = F.shape(target)
    new_shape = _get_merge_shape(src_dim, dst_dim, shape)
    new_target = mnp.moveaxis(target, src_dim, dst_dim)
    output = F.reshape(new_target, new_shape)
    return output, new_shape


@_primexpr
def _get_expand_shape(src_dim, dst_size, shape, prim_name):
    """Get new shape for splitting src_dim into dst_size parts."""
    dst_size2 = shape[src_dim] // dst_size
    new_shape = list(shape)
    new_shape[src_dim:(src_dim + 1)] = [dst_size, dst_size2]
    return tuple(new_shape)


def _reshape_expand_dims(src_dim, dst_size, target, prim_name):
    """Reshape target by splitting src_dim into dst_size parts."""
    shape = F.shape(target)
    new_shape = _get_expand_shape(src_dim, dst_size, shape, prim_name)
    return F.reshape(target, new_shape)


@_primexpr
def _get_new_size_by_index(input_size, batch_size, index):
    """Get the new size of input_size by multiplying input_size[index] by batch_size."""
    new_size = ()
    if input_size is None:
        return new_size
    new_size = list(input_size)
    new_size[index] *= batch_size
    return tuple(new_size)


@_primexpr
def _update_group_attr(prim, groups, batch_size):
    """Set new value for 'group' attribute of the convolution primitive."""
    group = groups * batch_size
    _vmap_update_prim_attr(prim, 'group', group)
    _vmap_update_prim_attr(prim, 'groups', group)


@constexpr
def _get_channel_index(data_format, prim_name):
    """Get channel index by data_format, only supports NHWC/NCHW/NCDHW now."""
    index = 0
    if data_format == "NHWC":
        index = 3
    elif data_format in ("NCHW", "NCDHW"):
        index = 1
    else:
        _raise_value_error("'data_format' in {} should be NHWC/NCHW/NCDHW, "
                           "but got {}.".format(prim_name, data_format))
    return index


def _conv_vmap_rule(prim, batch_size, input_bdim, weight_bdim, attr_list):
    """Vmap rule for Convolution operations, such as `Conv2D` and `Conv3D`."""
    input_x, x_dim = input_bdim
    weight, w_dim = weight_bdim
    prim_name = attr_list[0]
    groups = attr_list[1]
    data_format = attr_list[2]
    c_axis = _get_channel_index(data_format, prim_name)

    def _get_output_for_x_w_vmap():
        new_input, _ = _reshape_merge_dims(x_dim, c_axis, input_x)
        new_weight, new_w_shape = _reshape_merge_dims(w_dim, 0, weight)

        _update_group_attr(prim, groups, batch_size)
        _vmap_update_prim_attr(prim, 'out_channel', new_w_shape[0])
        out = prim(new_input, new_weight)
        out = _reshape_expand_dims(c_axis, batch_size, out, prim_name)
        return out, c_axis

    def _get_output_for_x_vmap():
        new_input, _ = _reshape_merge_dims(x_dim, 0, input_x)
        out = prim(new_input, weight)
        out = _reshape_expand_dims(0, batch_size, out, prim_name)
        return out, 0

    def _get_output_for_w_vmap():
        if groups > 1:
            expand_dim, merge_dim = _get_reshape_src_dim(w_dim, 0)
            new_weight = _reshape_expand_dims(expand_dim, groups, weight, prim_name)
            new_weight, _ = _reshape_merge_dims(merge_dim, 1, new_weight)
            new_weight, new_w_shape = _reshape_merge_dims(0, 0, new_weight)

            _vmap_update_prim_attr(prim, 'out_channel', new_w_shape[0])
            out = prim(input_x, new_weight)

            out = _reshape_expand_dims(c_axis, groups, out, prim_name)
            out = _reshape_expand_dims(c_axis + 1, batch_size, out, prim_name)
            out, _ = _reshape_merge_dims(c_axis, c_axis + 1, out)
            return out, c_axis

        new_weight, new_w_shape = _reshape_merge_dims(w_dim, 0, weight)
        _vmap_update_prim_attr(prim, 'out_channel', new_w_shape[0])
        out = prim(input_x, new_weight)
        out = _reshape_expand_dims(c_axis, batch_size, out, prim_name)
        return out, c_axis

    if x_dim is not None and w_dim is not None:
        if prim_name == "Conv3D":
            _raise_value_error("vmap in_axes of 'x' and 'weight in `{}` cannot be non-None at the same time,"
                               "but got {} and {}.".format(prim_name, x_dim, w_dim))
        output = _get_output_for_x_w_vmap()
    elif x_dim is not None:
        output = _get_output_for_x_vmap()
    else:
        output = _get_output_for_w_vmap()
    return output


def _conv_transpose_vmap_rule(prim, batch_size, dout_bdim, weight_bdim, input_size_bdim, attr_list):
    """
    Vmap rule for transposed convolution operations, such as `Conv2DTranspose`,
    `Conv2DBackpropInput`, `Conv3DTranspose` and `Conv3DBackpropInput`.
    """
    prim_name = attr_list[0]
    input_size = None
    if input_size_bdim is not None:
        input_size, input_size_dim = input_size_bdim
        if input_size_dim is not None:
            _raise_value_error("Vmap in_axes of 'input_size' in `{}` must be None, "
                               "but got {}.".format(prim_name, input_size_dim))
        if not isinstance(input_size, tuple):
            _raise_value_error("Unsupported vmap for dynamic shape of `{}` when "
                               "'input_size' is a tensor.".format(prim_name))

    dout, dout_dim = dout_bdim
    weight, w_dim = weight_bdim

    groups = attr_list[1]
    data_format = attr_list[2]
    c_axis = _get_channel_index(data_format, prim_name)

    def _get_conv_transpose_output(dout, weight, input_size):
        out = None
        if prim_name in ('Conv2DTranspose', 'Conv2DBackpropInput'):
            out = prim(dout, weight, input_size)
        elif prim_name == "Conv3DTranspose":
            out = prim(dout, weight)
        elif prim_name == "Conv3DBackpropInput":
            out = prim(weight, dout, input_size)
        else:
            _raise_value_error("Unsupported the operation: `{}`.".format(prim_name))
        return out

    def _get_output_for_dout_weight_vmap():
        _update_group_attr(prim, groups, batch_size)
        new_dout, _ = _reshape_merge_dims(dout_dim, c_axis, dout)
        new_weight, _ = _reshape_merge_dims(w_dim, 0, weight)
        new_input_size = _get_new_size_by_index(input_size, batch_size, c_axis)

        out = _get_conv_transpose_output(new_dout, new_weight, new_input_size)
        out = _reshape_expand_dims(c_axis, batch_size, out, prim_name)
        return out, c_axis

    def _get_output_for_dout_vmap():
        new_dout, _ = _reshape_merge_dims(dout_dim, 0, dout)
        new_input_size = _get_new_size_by_index(input_size, batch_size, 0)

        out = _get_conv_transpose_output(new_dout, weight, new_input_size)
        out = _reshape_expand_dims(0, batch_size, out, prim_name)
        return out, 0

    def _get_output_for_weight_vmap():
        new_weight, _ = _reshape_merge_dims(w_dim, c_axis, weight)
        new_input_size = _get_new_size_by_index(input_size, batch_size, c_axis)

        out = _get_conv_transpose_output(dout, new_weight, new_input_size)

        if groups > 1:
            out = _reshape_expand_dims(c_axis, groups, out, prim_name)
            out = _reshape_expand_dims(c_axis + 1, batch_size, out, prim_name)
            out, _ = _reshape_merge_dims(c_axis, c_axis + 1, out)
        else:
            out = _reshape_expand_dims(c_axis, batch_size, out, prim_name)
        return out, c_axis

    if dout_dim is not None and w_dim is not None:
        if prim_name in ("Conv3DTranspose", "Conv3DBackpropInput"):
            _raise_value_error("vmap in_axes of 'dout' and 'weight' in `{}` cannot be non-None at the same time,"
                               "but got {} and {}.".format(prim_name, dout_dim, w_dim))
        output = _get_output_for_dout_weight_vmap()
    elif dout_dim is not None:
        output = _get_output_for_dout_vmap()
    else:
        output = _get_output_for_weight_vmap()
    return output


def _conv_backprop_filter_vmap_rule(prim, batch_size, dout_bdim, input_bdim, weight_size_bdim, attr_list):
    """Vmap rule for `Conv2DBackpropFilter` and `Conv3DBackpropFilter` operations"""
    dout, dout_dim = dout_bdim
    input_x, x_dim = input_bdim
    weight_size, w_size_dim = weight_size_bdim

    prim_name = attr_list[0]
    groups = attr_list[1]
    data_format = attr_list[2]
    c_axis = _get_channel_index(data_format, prim_name)

    if w_size_dim is not None:
        _raise_value_error("Vmap in_axes of 'weight_size' in `{}` must be None, "
                           "but got {}.".format(prim_name, w_size_dim))
    if not isinstance(weight_size, tuple):
        _raise_value_error("Unsupported vmap for dynamic shape of `{}` when "
                           "'weight_size' is a tensor.".format(prim_name))

    def _get_conv_backprop_filter_output(dout, x, weight_size):
        out = None
        if prim_name == "Conv2DBackpropFilter":
            out = prim(dout, x, weight_size)
        elif prim_name == "Conv3DBackpropFilter":
            out = prim(x, dout, weight_size)
        else:
            _raise_value_error("Unsupported the operation: `{}`.".format(prim_name))
        return out

    def _get_output_for_dout_x_vmap():
        _update_group_attr(prim, groups, batch_size)

        new_dout, _ = _reshape_merge_dims(dout_dim, c_axis, dout)
        new_input, _ = _reshape_merge_dims(x_dim, c_axis, input_x)
        new_w_size = _get_new_size_by_index(weight_size, batch_size, 0)

        out = _get_conv_backprop_filter_output(new_dout, new_input, new_w_size)
        out = _reshape_expand_dims(0, batch_size, out, prim_name)
        return out, 0

    def _get_output_for_x_vmap():
        new_w_size = _get_new_size_by_index(weight_size, batch_size, c_axis)
        if groups > 1:
            expand_dim, merge_dim = _get_reshape_src_dim(x_dim, c_axis)
            new_input = _reshape_expand_dims(expand_dim, groups, input_x, prim_name)
            new_input, _ = _reshape_merge_dims(merge_dim, c_axis + 1, new_input)
            new_input, _ = _reshape_merge_dims(c_axis, c_axis, new_input)
        else:
            new_input, _ = _reshape_merge_dims(x_dim, c_axis, input_x)

        out = _get_conv_backprop_filter_output(dout, new_input, new_w_size)
        out = _reshape_expand_dims(c_axis, batch_size, out, prim_name)
        return out, c_axis

    def _get_output_for_dout_vmap():
        new_w_size = _get_new_size_by_index(weight_size, batch_size, 0)
        if groups > 1:
            expand_dim, merge_dim = _get_reshape_src_dim(dout_dim, c_axis)
            new_dout = _reshape_expand_dims(expand_dim, groups, dout, prim_name)
            new_dout, _ = _reshape_merge_dims(merge_dim, c_axis + 1, new_dout)
            new_dout, _ = _reshape_merge_dims(c_axis, c_axis, new_dout)

            out = _get_conv_backprop_filter_output(new_dout, input_x, new_w_size)
            out = _reshape_expand_dims(0, groups, out, prim_name)
            out = _reshape_expand_dims(1, batch_size, out, prim_name)
            out, _ = _reshape_merge_dims(0, 1, out)
            return out, 0

        new_dout, _ = _reshape_merge_dims(dout_dim, c_axis, dout)
        out = _get_conv_backprop_filter_output(new_dout, input_x, new_w_size)
        out = _reshape_expand_dims(0, batch_size, out, prim_name)
        return out, 0

    if dout_dim is not None and x_dim is not None:
        if prim_name == "Conv3DBackpropFilter":
            _raise_value_error("vmap in_axes of 'dout' and 'x' in `{}` cannot be non-None at the same time,"
                               "but got {} and {}.".format(prim_name, dout_dim, x_dim))
        output = _get_output_for_dout_x_vmap()
    elif x_dim is not None:
        output = _get_output_for_x_vmap()
    else:
        output = _get_output_for_dout_vmap()
    return output

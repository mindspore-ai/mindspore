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

"""nn_ops vmap impl."""

from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import constexpr
from ..primitive import Primitive
from .._vmap.vmap_base import vmap_rules_getters, vmap_general_preprocess, get_unop_vmap_rule, _bdim_at_front


@constexpr
def _get_bias_broadcast_shape(x_shape, bias_shape, bias_dim, data_format):
    """Get the broadcast shape for bias and use it in 'BiasAdd' VmapRule."""
    bias_rank = len(bias_shape)
    if bias_dim is None and bias_rank == 1:
        bias_batch = 1
        bias_channel = bias_shape[0]
    elif bias_dim is not None and bias_rank == 2:
        bias_batch = bias_shape[0]
        bias_channel = bias_shape[1]
    else:
        raise ValueError("The rank of 'bias' in 'BiasAdd' operator is invalid, which is rank: {}"
                         " with bias_dim: {}.".format(bias_rank, bias_dim))

    # The 'Biasadd' operator supports 2-5 dimensions input, and another 'batch' dimension is added to the front in
    # vmap scenario.
    x_min_rank = 3
    x_max_rank = 5
    if data_format == "NCDHW":
        x_max_rank += 1
    x_rank = len(x_shape)

    if x_rank < x_min_rank or x_rank > x_max_rank:
        raise ValueError("For primitive[BiasAdd] in vmap, the dims of input_x must be in [x_min_rank, {}"
                         "], but got {}.".format(x_max_rank, x_rank))

    if data_format == "NHWC":
        # In the 'NHWC' data format ('BN**C' actually), the last dimension is channel axis.
        x_channel = x_shape[-1]
        if x_channel != bias_channel:
            raise ValueError("For 'BiadAdd, bias_channel should be equal to x_channel, "
                             "but got date format: {}, got bias_channel: {}, "
                             "x_channel: {}.".format(data_format, bias_channel, x_channel))
        if bias_dim is None:
            bias_broadcast_shape = (1,) * (x_rank - bias_rank) + (bias_channel,)
        else:
            bias_broadcast_shape = (bias_batch,) + (1,) * (x_rank - bias_rank) + (bias_channel,)
    else:
        # In the 'NCHW' or 'NCDHW' data format ('BNC**' actually), the third dimension is channel axis.
        x_channel = x_shape[2]
        if x_channel != bias_channel:
            raise ValueError("For 'BiadAdd, bias_channel should be equal to x_channel, but got date format: "
                             "{}, got bias_channel: {}, x_channel: {}.".format(data_format, bias_channel, x_channel))
        bias_broadcast_shape = (bias_batch, 1, bias_channel)
        if x_rank == x_min_rank:
            return bias_broadcast_shape
        bias_broadcast_shape = bias_broadcast_shape + (1,) * (x_rank - x_min_rank)
    return bias_broadcast_shape


@vmap_rules_getters.register(P.BiasAdd)
def get_bias_add_vmap_rule(prim, axis_size):
    """VmapRule for `BiasAdd` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)
        data_format = "NCHW"
    else:
        data_format = prim.data_format
    add_op = P.Add()

    def vmap_rule(input_bdim, bias_bdim):
        is_all_none, result = vmap_general_preprocess(prim, input_bdim, bias_bdim)
        if is_all_none:
            return result

        input_x, x_dim = input_bdim
        bias, bias_dim = bias_bdim
        input_x = _bdim_at_front(input_x, x_dim, axis_size)
        if bias_dim is not None:
            bias = _bdim_at_front(bias, bias_dim, axis_size)
        x_shape = F.shape(input_x)
        bias_shape = F.shape(bias)
        bias_broadcast_shape = _get_bias_broadcast_shape(x_shape, bias_shape, bias_dim, data_format)
        bias = F.reshape(bias, bias_broadcast_shape)
        out = add_op(input_x, bias)
        return (out, 0)

    return vmap_rule


@vmap_rules_getters.register(P.Dropout2D)
@vmap_rules_getters.register(P.Dropout3D)
def get_dropout_nd_vmap_rule(prim, axis_size):
    """VmapRule for 'DropoutND' operation."""

    def vmap_rule(x_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        x = _bdim_at_front(x, x_dim, axis_size)
        output, mask = prim(x)
        return (output, 0), (mask, 0)

    return vmap_rule


get_unop_vmap_rule = vmap_rules_getters.register(P.Elu)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.ReLU)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.ReLU6)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.SeLU)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.HSigmoid)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Softplus)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.SoftShrink)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.HShrink)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.GeLU)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Gelu)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.FastGeLU)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.FastGelu)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.HSwish)(get_unop_vmap_rule)

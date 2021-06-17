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
"""Utils."""
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops.functional as F
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.communication.management import get_group_size
from mindspore.common.seed import _get_graph_seed
from mindspore._checkparam import Validator
from mindspore import context

from src.weight_init import normal_weight, zero_weight, one_weight


class ResidualConnection(nn.Cell):
    """
    Add residual to output.

    Args:
        dropout_prob (float): Dropout rate.

    Returns:
        Tensor, with the same shape of hidden_tensor
    """

    def __init__(self, dropout_prob=0.0):
        super(ResidualConnection, self).__init__()
        self.add = P.TensorAdd()

    def construct(self, hidden_tensor, input_tensor):
        # hidden_tensor is the output of sublayer
        output = hidden_tensor
        output = self.add(output, input_tensor)
        return output


class LinearLayer(nn.Cell):
    """
    Args:
        input_size (int): The number of input features.
        output_size (int): The number of output features.
    """

    def __init__(self,
                 input_size,
                 output_size):
        super(LinearLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = Parameter(normal_weight([output_size, input_size], output_size), name='projection_weight')
        self.bias = Parameter(zero_weight(output_size), name='projection_bias')
        self.matmul = P.MatMul(transpose_b=True)
        self.bias_add = P.BiasAdd()
        self.cast = P.Cast()

    def construct(self, input_tensor):
        input_tensor = self.cast(input_tensor, mstype.float16)
        fp16_weight = self.cast(self.weight, mstype.float16)

        output_tensor = self.matmul(input_tensor, fp16_weight)
        fp16_bias = self.cast(self.bias, mstype.float16)
        output_tensor = self.bias_add(output_tensor, fp16_bias)
        output_tensor = self.cast(output_tensor, mstype.float32)
        return output_tensor


class LayerNorm(nn.Cell):
    r"""
        A self-defined layer norm operation using reduce sum and reduce mean
    """

    def __init__(self, normalized_shape, config=None, epsilon=1e-5, scale=1e-3):
        super(LayerNorm, self).__init__()
        self.gamma = Parameter(one_weight(normalized_shape, mstype.float32), name="gamma")
        self.beta = Parameter(zero_weight(normalized_shape, mstype.float32), name="beta")
        self.mean = P.ReduceMean(keep_dims=True).shard(((config.dp, 1),))
        self.square = P.Square().shard(((config.dp, 1),))
        self.sqrt = P.Sqrt().shard(((config.dp, 1),))
        self.sub1 = P.Sub().shard(((config.dp, 1), (config.dp, 1)))
        self.add = P.TensorAdd().shard(((config.dp, 1), ()))
        self.eps = epsilon
        self.mul = P.Mul().shard(((config.dp, 1), (1,)))
        self.add2 = P.TensorAdd().shard(((config.dp, 1), (1,)))
        self.real_div = P.RealDiv().shard(((config.dp, 1), (config.dp, 1)))

    def construct(self, x):
        mean = self.mean(x, -1)
        diff = self.sub1(x, mean)
        variance = self.mean(self.square(diff), -1)
        variance_eps = self.sqrt(self.add(variance, self.eps))
        output = self.real_div(diff, variance_eps)
        output = self.add2(self.mul(output, self.gamma), self.beta)
        return output


get_square_sum = C.MultitypeFuncGraph("get_square_sum")

@get_square_sum.register("Tensor", "Tensor")
def _get_square_sum(grad, value):
    norm = P.ReduceSum(False)(F.square(grad) / value, ())
    norm = F.expand_dims(F.cast(norm, mstype.float32), 0)
    return norm

@get_square_sum.register("Tensor", "Number")
def _get_square_sum_number(grad, value):
    norm = P.ReduceSum(False)(F.square(grad), ()) / value
    norm = F.expand_dims(F.cast(norm, mstype.float32), 0)
    return norm


apply_global_norm = C.MultitypeFuncGraph("apply_global_norm")

@apply_global_norm.register("Tensor", "Tensor", "Tensor")
def _apply_global_norm(clip_norm, global_norm, grad):
    grad = grad * clip_norm / global_norm
    return grad


class GlobalNorm(nn.Cell):
    r"""
        Calculate the global norm value of given tensors
    """

    def __init__(self, params):
        super(GlobalNorm, self).__init__()
        self.norm = nn.Norm()
        self.hyper_map = C.HyperMap()

        self.allreduce_filter = tuple(
            "layernorm" not in x.name and
            "dense_proj.bias" not in x.name and
            "embedding_table" not in x.name for x in params)

        self.values = []
        self.group_size = get_group_size()
        for item in self.allreduce_filter:
            if item:
                self.values.append(Tensor([1.0], mstype.float32))
            else:
                self.values.append(Tensor([self.group_size * 1.0], mstype.float32))
        self.values = tuple(self.values)

    def construct(self, grads):
        square_sum_dp = self.hyper_map(get_square_sum, grads, self.values)
        global_norms = F.sqrt(P.AllReduce()(F.addn(square_sum_dp)))
        return global_norms


class ClipByGlobalNorm(nn.Cell):
    r"""
        Clip grads by global norm
    """

    def __init__(self, params, clip_norm=1.0):
        super(ClipByGlobalNorm, self).__init__()
        self.global_norm = GlobalNorm(params)
        self.clip_norm = Tensor([clip_norm], mstype.float32)
        self.hyper_map = C.HyperMap()

    def construct(self, grads):
        global_norm_value = self.global_norm(grads)
        cond = P.GreaterEqual()(global_norm_value, self.clip_norm)
        # P.Print()("do clip ", cond, ", global norm is ", global_norm_value)
        global_norm = F.select(cond, global_norm_value, self.clip_norm)
        grads = self.hyper_map(F.partial(apply_global_norm, self.clip_norm, global_norm), grads)
        return grads, global_norm_value


class Dropout(nn.Cell):
    r"""
        A Dropout Implements with P.DropoutGenMask and  P.DropoutDoMask for parallel training.
    """

    def __init__(self, keep_prob=0.5, dtype=mstype.float32):
        super(Dropout, self).__init__()
        if keep_prob <= 0 or keep_prob > 1:
            raise ValueError("dropout probability should be a number in range (0, 1], but got {}".format(keep_prob))
        Validator.check_subclass("dtype", dtype, mstype.number_type, self.cls_name)
        Validator.check_value_type('keep_prob', keep_prob, [float], self.cls_name)
        self.keep_prob = keep_prob
        seed0, seed1 = _get_graph_seed(0, "dropout")
        self.seed0 = seed0
        self.seed1 = seed1
        self.dtype = dtype
        self.get_shape = P.Shape()
        self.dropout_gen_mask = P.DropoutGenMask(Seed0=self.seed0, Seed1=self.seed1)
        self.dropout_do_mask = P.DropoutDoMask()
        self.cast = P.Cast()
        self.is_ascend = context.get_context('device_target') in ["Ascend"]
        self.dropout = P.Dropout(keep_prob)

    def construct(self, x):
        r"""
        Input: a tensor
        Returns: a tensor
        """
        if not self.training:
            return x

        if not self.is_ascend:
            out, _ = self.dropout(x)
            return out

        if self.keep_prob == 1:
            return x

        shape = self.get_shape(x)
        dtype = P.DType()(x)
        keep_prob = self.cast(self.keep_prob, dtype)
        output = self.dropout_gen_mask(shape, keep_prob)
        return self.dropout_do_mask(x, output, keep_prob)

    def extend_repr(self):
        return 'keep_prob={}, dtype={}'.format(self.keep_prob, self.dtype)

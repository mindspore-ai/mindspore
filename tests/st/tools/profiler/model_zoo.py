# Copyright 2024 Huawei Technologies Co., Ltd
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
import numpy as np
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore import nn, Parameter, Tensor
from mindspore.common.initializer import initializer
from mindspore.common.initializer import TruncatedNormal


class TinyAddNet(nn.Cell):
    def __init__(self):
        super(TinyAddNet, self).__init__()
        self.add = P.Add()

    def construct(self, x_, y_):
        return self.add(x_, y_)


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """weight initial for conv layer"""
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    """weight initial for fc layer"""
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


def weight_variable():
    """weight initial"""
    return TruncatedNormal(0.02)


class FusedRMSNorm(nn.Cell):
    r"""
    A RMSNorm fused kernel implementation.

    Args:
        dim (tuple): The shape of the input tensor
        eps (float): The epsilon value of the denominator. Default 1e-5.
        param_init_type: The param init type.

    Inputs:
        - **x** (Tensor) - Tensor of shape (batch, seq_length, hidden_size).

    Outputs:
        - Tensor with shape (batch, seq_length, hidden_size).
    """

    def __init__(self, dim, eps=1.e-6, compute_type=mstype.float32):
        super(FusedRMSNorm, self).__init__()
        self.eps = eps
        self.compute_type = compute_type
        self.weight = Parameter(initializer("ones", (dim,), dtype=mstype.float32), parallel_optimizer=False)

        self.norm = P.RmsNorm(eps)
        self.cast = P.Cast()
        self.rcast = P.Cast()

    def construct(self, x):
        """Forward of FusedRMSNorm."""
        original_type = x.dtype
        output = self.norm(self.cast(x, self.compute_type), self.weight)[0]
        return self.rcast(output, original_type)


class LeNet5(nn.Cell):
    """Define LeNet5 network."""

    def __init__(self, num_class=10, channel=1):
        """Net init."""
        super(LeNet5, self).__init__()
        self.num_class = num_class
        self.conv1 = conv(channel, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, self.num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.channel = Tensor(channel)

    def construct(self, data):
        """define construct."""
        output = self.conv1(data)
        output = self.relu(output)
        output = self.max_pool2d(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.max_pool2d(output)
        output = self.flatten(output)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        return output


class LeNet5WithRMSNorm(nn.Cell):
    """Define LeNet5 network with RMSNorm."""

    def __init__(self, num_class=10, channel=1):
        super(LeNet5WithRMSNorm, self).__init__()
        self.lenent5 = LeNet5(num_class, channel)
        self.norm = FusedRMSNorm(dim=num_class, eps=1e-6, compute_type=mstype.float32)

    def construct(self, data):
        out = self.lenent5(data)
        out = self.norm(out)

        return out


class TinyTransformer(nn.Cell):
    """ Tiny Transformer for light profiling"""
    def __init__(self,
                 d_model=32,
                 nhead=2,
                 num_encoder_layers=1,
                 num_decoder_layers=1,
                 dim_feedforward=64
                 ):
        super(TinyTransformer, self).__init__()
        self.model = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward
        )

    def construct(self, src, tgt):
        return self.model(src, tgt)


class DynamicShapeNet(nn.Cell):
    def __init__(self):
        super(DynamicShapeNet, self).__init__()
        self.unique = P.Unique()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.add = P.Add()

    def construct(self, a, b):
        val = self.add(a, b)
        size = self.shape(val)
        res = self.reshape(val, size)
        return res


class CustomAICpuNet(nn.Cell):
    """custom ai cpu net"""
    def __init__(self):
        super(CustomAICpuNet, self).__init__()
        self.select = P.Select()
        self.reshape = P.Reshape()
        self.xlogy = P.Xlogy()
        self.tril = P.Tril(10)
        self.cast = P.Cast()
        self.expand_dims = P.ExpandDims()
        self.dense = nn.Dense(1, 3, activation='relu')
        self.flatten = nn.Flatten()

    def construct(self, a):
        shape = (2, 3)
        b = Tensor(np.array([4, 4, 5, 5, 6, 6]), mstype.float64)
        input_cond = Tensor([True, False, True, False, True, False])
        a = self.select(input_cond, a, b)
        a = self.reshape(a, shape)
        b = self.reshape(b, shape)
        a = self.tril(a)

        output = self.xlogy(a, b)
        output = self.expand_dims(output, -1)
        output = self.cast(output, mstype.float32)
        output = self.dense(output)
        output = self.flatten(output)
        return output


class AllReduceNet(nn.Cell):
    def __init__(self):
        super(AllReduceNet, self).__init__()
        self.mul = P.Mul()
        self.all_reduce = P.AllReduce()
        self.add = P.Add()
        self.y1 = Tensor(np.array([[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]])).astype(np.float32)
        self.y2 = Tensor(np.array([[-16, -16, -16, -16], [-16, -16, -16, -16], \
                                   [-16, -16, -16, -16]])).astype(np.float32)

    def construct(self, x):
        x = self.mul(x, 2)
        z = self.add(x, self.y1)
        z = self.all_reduce(z)
        out = self.add(z, self.y2)
        out = self.all_reduce(out)
        out = self.mul(out, 2)
        return out

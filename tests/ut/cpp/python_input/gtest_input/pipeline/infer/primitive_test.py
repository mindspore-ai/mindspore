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
import mindspore.nn as nn
from mindspore.common import dtype
from mindspore.ops import operations as P
from mindspore.ops import prim_attr_register, PrimitiveWithInfer


def get_add(a, b):
    return a + b


def get_f(v):
    return v + 1


relu = nn.ReLU()


def get_relu(x):
    return relu(x)


softmax_cross_entropy_with_logits = P.SoftmaxCrossEntropyWithLogits()


def get_softmax_cross_entropy_with_logits(logits, labels):
    return softmax_cross_entropy_with_logits(logits, labels)


class TensorToScalar(PrimitiveWithInfer):
    """this is a test primitive for cases that has tensor input, but has only one scalar output"""

    @prim_attr_register
    def __init__(self):
        """init"""

    def __call__(self, logits, labels):
        raise NotImplementedError

    def infer_shape(self, logits_shape, label_shape):
        return []

    def infer_dtype(self, logits_type, labels_type):
        # pylint: disable=unused-argument
        return dtype.float64


tensorToScalar = TensorToScalar()


def get_tensor_to_scalar(logits, labels):
    return tensorToScalar(logits, labels)


conv2d = P.Conv2D(64,
                  (3, 3),
                  pad_mode="pad",
                  pad=1,
                  stride=2)


def get_conv2d(x, w):
    return conv2d(x, w)


conv2dNative = P.DepthwiseConv2dNative(3, (3, 3), pad_mode="pad", pad=1, stride=2)


def get_conv2d_native(x, w):
    return conv2dNative(x, w)


biasAdd = P.BiasAdd()


def get_bias_add(x, b):
    return biasAdd(x, b)


def test_conv2d(out_channel, kernel_size, pad, stride, dilation):
    conv = P.Conv2D(out_channel=out_channel, kernel_size=kernel_size, pad_mode="pad", pad=pad,
                    stride=stride, dilation=dilation)

    def get_conv(x, w):
        return conv(x, w)

    return get_conv


def test_dropout():
    dropOutGenMask = P.DropoutGenMask()
    dropoutDoMask = P.DropoutDoMask()
    shape = P.Shape()

    def get_dropout(x, prob):
        mask = dropOutGenMask(shape(x), prob)
        y = dropoutDoMask(x, mask, prob)
        return y

    return get_dropout

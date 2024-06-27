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
import numpy as np
from tests.mark_utils import arg_mark
import mindspore.context as context
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self, input_scale, input_bias, input_mean, input_variance, is_training):
        super(Net, self).__init__()
        self.fused_bn_ex = P.BatchNorm(is_training=is_training, epsilon=1e-5, momentum=0.9)
        self.scale = Parameter(input_scale, name='scale')
        self.bias = Parameter(input_bias, name='b')
        self.mean = Parameter(input_mean, name='mean')
        self.variance = Parameter(input_variance, name='variance')

    def construct(self, input_x):
        return self.fused_bn_ex(input_x, self.scale, self.bias, self.mean, self.variance)


def get_output(x, weight, bias, moving_mean, moving_var, is_training, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    net = Net(Tensor(weight), Tensor(bias), Tensor(moving_mean), Tensor(moving_var), is_training)
    output = net(Tensor(x))
    return output, net.mean, net.variance


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bn_train():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = np.random.normal(0, 1, [1, 2, 4, 4]).astype(np.float32)
    weight = np.random.normal(0, 1, [2]).astype(np.float32)
    bias = np.random.normal(0, 1, [2]).astype(np.float32)
    moving_mean = np.random.normal(0, 1, [2]).astype(np.float32)
    moving_var = np.random.normal(0, 1, [2]).astype(np.float32)

    train_expect = get_output(x, weight, bias, moving_mean, moving_var, True, False)
    train_output = get_output(x, weight, bias, moving_mean, moving_var, True, True)

    assert np.allclose(train_expect[0][0].asnumpy(), train_output[0][0].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(train_expect[0][3].asnumpy(), train_output[0][3].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(train_expect[0][4].asnumpy(), train_output[0][4].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(train_expect[1].data.asnumpy(), train_output[1].data.asnumpy(), 0.0001, 0.0001)
    assert np.allclose(train_expect[2].data.asnumpy(), train_output[2].data.asnumpy(), 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bn_infer():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(mode=context.GRAPH_MODE, graph_kernel_flags='--enable_expand_ops=BatchNorm')
    x = np.random.normal(5, 1, [1, 2, 4, 4]).astype(np.float32)
    weight = np.random.normal(5, 1, [2]).astype(np.float32)
    bias = np.random.normal(5, 1, [2]).astype(np.float32)
    moving_mean = np.random.normal(5, 1, [2]).astype(np.float32)
    moving_var = np.random.normal(5, 1, [2]).astype(np.float32)

    infer_expect = get_output(x, weight, bias, moving_mean, moving_var, False, False)
    infer_output = get_output(x, weight, bias, moving_mean, moving_var, False, True)

    assert np.allclose(infer_expect[0][0].asnumpy(), infer_output[0][0].asnumpy(), 0.0001, 0.0001)

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
from mindspore.ops.operations import _grad_ops as G


class Net(nn.Cell):
    def __init__(self, is_training):
        super(Net, self).__init__()
        self.fused_bn_grad_ex = G.BatchNormGrad(is_training=is_training, epsilon=1e-5)

    def construct(self, input_dy, input_x, input_scale, input_save_mean, input_save_inv_variance, input_reverse):
        return self.fused_bn_grad_ex(
            input_dy, input_x, input_scale, input_save_mean, input_save_inv_variance, input_reverse)


def get_output(input_dy, input_x, input_scale, input_save_mean, input_save_inv_variance, input_reverse,
               is_training, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    net = Net(is_training)
    output = net(input_dy, input_x, input_scale, input_save_mean, input_save_inv_variance, input_reverse)
    return output


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_bn_grad_train():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(mode=context.GRAPH_MODE, graph_kernel_flags='--enable_expand_ops=BatchNormGrad')
    input_dy = Tensor(np.random.normal(5, 1, [1, 2, 4, 4]).astype(np.float32))
    input_x = Tensor(np.random.normal(5, 1, [1, 2, 4, 4]).astype(np.float32))
    input_scale = Tensor(np.random.normal(5, 1, [2]).astype(np.float32))
    input_save_mean = Tensor(np.random.normal(5, 1, [2]).astype(np.float32))
    input_save_inv_variance = Tensor(np.random.normal(5, 1, [2]).astype(np.float32))
    input_reverse = Tensor(np.random.normal(5, 1, [2]).astype(np.float32))

    expect = get_output(
        input_dy, input_x, input_scale, input_save_mean, input_save_inv_variance, input_reverse, True, False)
    output = get_output(
        input_dy, input_x, input_scale, input_save_mean, input_save_inv_variance, input_reverse, True, True)

    assert np.allclose(expect[0].asnumpy(), output[0].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(expect[1].asnumpy(), output[1].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(expect[2].asnumpy(), output[2].asnumpy(), 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bn_grad_infer():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(mode=context.GRAPH_MODE, graph_kernel_flags='--enable_expand_ops=BatchNormGrad')
    input_dy = Tensor(np.random.normal(5, 1, [1, 2, 4, 4]).astype(np.float32))
    input_x = Tensor(np.random.normal(5, 1, [1, 2, 4, 4]).astype(np.float32))
    input_scale = Tensor(np.random.normal(5, 1, [2]).astype(np.float32))
    input_save_mean = Tensor(np.random.normal(5, 1, [2]).astype(np.float32))
    input_save_inv_variance = Tensor(np.random.normal(5, 1, [2]).astype(np.float32))
    input_reverse = Tensor(np.random.normal(5, 1, [2]).astype(np.float32))

    expect = get_output(
        input_dy, input_x, input_scale, input_save_mean, input_save_inv_variance, input_reverse, False, False)
    output = get_output(
        input_dy, input_x, input_scale, input_save_mean, input_save_inv_variance, input_reverse, False, True)

    assert np.allclose(expect[0].asnumpy(), output[0].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(expect[1].asnumpy(), output[1].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(expect[2].asnumpy(), output[2].asnumpy(), 0.0001, 0.0001)

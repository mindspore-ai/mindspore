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
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.communication.management import init
from mindspore.ops import composite as C

# define target and input values here
x_fwd_input = np.array([[
        [[1, 3, 3, 5], [2, 4, 6, 8], [3, 6, 7, 7], [4, 3, 8, 2]],
        [[5, 7, 6, 3], [3, 5, 6, 7], [9, 4, 2, 5], [7, 5, 8, 1]]]]).astype(np.float32)
expect_output_fwd = np.array([[[[-0.6059, 0.3118, 0.3118, 1.2294],
                                [-0.1471, 0.7706, 1.6882, 2.6059],
                                [0.3118, 1.6882, 2.1471, 2.1471],
                                [0.7706, 0.3118, 2.6059, -0.1471]],
                               [[0.9119, 1.8518, 1.3819, -0.0281],
                                [-0.0281, 0.9119, 1.3819, 1.8518],
                                [2.7918, 0.4419, -0.4981, 0.9119],
                                [1.8518, 0.9119, 2.3218, -0.9680]]]]).astype(np.float32)
grad_back = np.array([[[[1, 2, 7, 1], [4, 2, 1, 3], [1, 6, 5, 2], [2, 4, 3, 2]],
                       [[9, 4, 3, 5], [1, 3, 7, 6], [5, 7, 9, 9], [1, 4, 6, 8]]]]).astype(np.float32)
expect_output_back = np.array([[[[-0.69126546, -0.32903028, 1.9651246, -0.88445705],
                                 [0.6369296, -0.37732816, -0.93275493, -0.11168876],
                                 [-0.7878612, 1.3614, 0.8542711, -0.52222186],
                                 [-0.37732816, 0.5886317, -0.11168876, -0.28073236]],
                                [[1.6447213, -0.38968924, -1.0174079, -0.55067265],
                                 [-2.4305856, -1.1751484, 0.86250514, 0.5502673],
                                 [0.39576983, 0.5470243, 1.1715001, 1.6447213],
                                 [-1.7996241, -0.7051701, 0.7080077, 0.5437813]]]]).astype(np.float32)

class Net(nn.Cell):
    def __init__(self, c):
        super(Net, self).__init__()
        self.num_features = c
        self.eps = 1e-5
        self.momentum = 1
        self.mode = True
        self.affine = True
        self.sync_bn_op = nn.SyncBatchNorm(num_features=self.num_features,
                                           eps=self.eps,
                                           momentum=self.momentum,
                                           affine=self.affine,
                                           gamma_init='ones',
                                           beta_init='ones',
                                           moving_mean_init='ones',
                                           moving_var_init='ones',
                                           use_batch_statistics=True,
                                           process_groups=None)
    def construct(self, input_data):
        return self.sync_bn_op(input_data)

class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_data, sens):
        gout = self.grad(self.network)(input_data, sens)
        return gout

def test_sync_batch_norm_forward_fp32_graph():
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    init()
    x = x_fwd_input.copy().astype(np.float32)
    expect_output = expect_output_fwd.copy().astype(np.float32)
    overall_shape = x.shape
    error = np.ones(shape=overall_shape) * 1.0e-4
    net = Net(2)
    net.set_train()
    output = net(Tensor(x))
    diff = output.asnumpy() - expect_output
    assert np.all(diff < error)
    assert np.all(-diff < error)

def test_sync_batch_norm_forward_fp16_pynative():
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    init()
    x = x_fwd_input.copy().astype(np.float16)
    expect_output = expect_output_fwd.copy().astype(np.float16)
    overall_shape = x.shape
    error = np.ones(shape=overall_shape) * 1.0e-3
    net = Net(2)
    net.set_train()
    output = net(Tensor(x))
    diff = output.asnumpy() - expect_output
    assert np.all(diff < error)
    assert np.all(-diff < error)

def test_sync_batch_norm_backwards_fp32_graph():
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    init()
    x = x_fwd_input.copy().astype(np.float32)
    expect_output = expect_output_back.copy().astype(np.float32)
    grad = grad_back.copy().astype(np.float32)
    overall_shape = x.shape
    error = np.ones(shape=overall_shape) * 1.0e-5
    fwd_net = Net(2)
    fwd_net.set_train()
    bn_grad = Grad(fwd_net)
    output = bn_grad(Tensor(x), Tensor(grad))
    diff = output[0].asnumpy() - expect_output
    assert np.all(diff < error)
    assert np.all(-diff < error)

def test_sync_batch_norm_backwards_fp16_pynative():
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    init()
    x = x_fwd_input.copy().astype(np.float16)
    expect_output = expect_output_back.copy().astype(np.float16)
    grad = grad_back.copy().astype(np.float16)
    overall_shape = x.shape
    error = np.ones(shape=overall_shape) * 1.0e-3
    fwd_net = Net(2)
    fwd_net.set_train()
    bn_grad = Grad(fwd_net)
    output = bn_grad(Tensor(x), Tensor(grad))
    diff = output[0].asnumpy() - expect_output
    assert np.all(diff < error)
    assert np.all(-diff < error)

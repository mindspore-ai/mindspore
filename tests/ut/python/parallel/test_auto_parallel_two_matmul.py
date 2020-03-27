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

import numpy as np
from mindspore import context
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor
from tests.ut.python.ops.test_math_ops import VirtualLoss
import mindspore as ms
from mindspore.common.api import _executor
from mindspore.ops import composite as C
from mindspore.parallel import _cost_model_context as cost_model_context
from mindspore.parallel import set_algo_parameters, get_algo_parameters, reset_algo_parameters
from mindspore.parallel._utils import _reset_op_id as reset_op_id

class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y, b):
        predict = self.network(x, y, b)
        return self.loss(predict)

class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y, b):
        return C.grad_all(self.network)(x, y, b)

    # model_parallel test
def test_two_matmul():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.matmul1 = P.MatMul()
            self.matmul2 = P.MatMul()

        def construct(self, x, y, b):
            out = self.matmul1(x, y)
            out = self.matmul2(out, b)
            return out

    size = 16
    context.set_auto_parallel_context(device_num=size, global_rank=0)
    cost_model_context.set_cost_model_context(device_memory_capacity= 32.0 * 1024.0 * 1024.0 * 1024.0,
                                              costmodel_alpha=1.0,
                                              costmodel_beta=60.0,
                                              costmodel_gamma=0.1,
                                              costmodel_communi_threshold=1024.0,
                                              costmodel_communi_const=2222.0,
                                              costmodel_communi_bias=1111.0)
    dev_mem_cap = cost_model_context.get_cost_model_context("device_memory_capacity")
    assert dev_mem_cap == 32.0 * 1024.0 * 1024.0 * 1024.0
    costmodel_alpha = cost_model_context.get_cost_model_context("costmodel_alpha")
    assert costmodel_alpha == 1.0
    costmodel_beta = cost_model_context.get_cost_model_context("costmodel_beta")
    assert costmodel_beta == 60.0
    costmodel_gamma = cost_model_context.get_cost_model_context("costmodel_gamma")
    assert costmodel_gamma == 0.1
    costmodel_communi_threshold = cost_model_context.get_cost_model_context("costmodel_communi_threshold")
    assert costmodel_communi_threshold == 1024.0
    costmodel_communi_const = cost_model_context.get_cost_model_context("costmodel_communi_const")
    assert costmodel_communi_const == 2222.0
    costmodel_communi_bias = cost_model_context.get_cost_model_context("costmodel_communi_bias")
    assert costmodel_communi_bias == 1111.0

    cost_model_context.reset_cost_model_context()
    dev_mem_cap = cost_model_context.get_cost_model_context("device_memory_capacity")
    assert dev_mem_cap == 16.0 * 1024.0 * 1024.0 * 1024.0
    costmodel_alpha = cost_model_context.get_cost_model_context("costmodel_alpha")
    assert costmodel_alpha == 1.0
    costmodel_beta = cost_model_context.get_cost_model_context("costmodel_beta")
    assert costmodel_beta == 65.0
    costmodel_gamma = cost_model_context.get_cost_model_context("costmodel_gamma")
    assert costmodel_gamma == 0.02
    costmodel_communi_threshold = cost_model_context.get_cost_model_context("costmodel_communi_threshold")
    assert costmodel_communi_threshold == 2048.0
    costmodel_communi_const = cost_model_context.get_cost_model_context("costmodel_communi_const")
    assert costmodel_communi_const == 3072.0
    costmodel_communi_bias = cost_model_context.get_cost_model_context("costmodel_communi_bias")
    assert costmodel_communi_bias == 1024.0


    set_algo_parameters(simplify_cal=True,
                                          tensor_slice_align_enable=False,
                                          tensor_slice_align_size=32,
                                          not_fully_use_devices=True,
                                          elementwise_op_strategy_follow=False)
    para_simplify_cal = get_algo_parameters("simplify_cal")
    assert para_simplify_cal == True
    para_slice_align_enable = get_algo_parameters("tensor_slice_align_enable")
    assert para_slice_align_enable == False
    para_slice_align_size = get_algo_parameters("tensor_slice_align_size")
    assert para_slice_align_size == 32
    not_fully_use_devices  = get_algo_parameters("not_fully_use_devices")
    assert not_fully_use_devices == True
    elementwise_op_strategy_follow = get_algo_parameters("elementwise_op_strategy_follow")
    assert elementwise_op_strategy_follow == False

    reset_algo_parameters()
    para_simplify_cal = get_algo_parameters("simplify_cal")
    assert para_simplify_cal == True
    para_slice_align_enable = get_algo_parameters("tensor_slice_align_enable")
    assert para_slice_align_enable == False
    para_slice_align_size = get_algo_parameters("tensor_slice_align_size")
    assert para_slice_align_size == 16
    not_fully_use_devices  = get_algo_parameters("not_fully_use_devices")
    assert not_fully_use_devices == False
    elementwise_op_strategy_follow = get_algo_parameters("elementwise_op_strategy_follow")
    assert elementwise_op_strategy_follow == False

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)

    net = NetWithLoss(Net())
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    reset_op_id()
    
    _executor.compile(net, x, y, b, phase='train')
    strategies = _executor._get_strategy(net)
    expected_strategies = {'Default/network-Net/MatMul-op2': [[16, 1], [1, 1]],
                     'Default/network-Net/MatMul-op3': [[16, 1], [1, 1]]}
    assert strategies == expected_strategies
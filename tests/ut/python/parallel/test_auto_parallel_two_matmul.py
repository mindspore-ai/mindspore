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

import re
import math
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.parallel import _cost_model_context as cost_model_context
from mindspore.parallel._cost_model_context import _set_algo_single_loop, _get_algo_single_loop
from mindspore.parallel import set_algo_parameters, get_algo_parameters, reset_algo_parameters
from mindspore.parallel._utils import _reset_op_id as reset_op_id
from tests.ut.python.ops.test_math_ops import VirtualLoss


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


grad_all = C.GradOperation(get_all=True)


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
        return grad_all(self.network)(x, y, b)

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
    cost_model_context.set_cost_model_context(device_memory_capacity=32.0 * 1024.0 * 1024.0 * 1024.0,
                                              costmodel_alpha=1.0,
                                              costmodel_beta=60.0,
                                              costmodel_gamma=0.1,
                                              costmodel_communi_threshold=1024.0,
                                              costmodel_communi_const=2222.0,
                                              costmodel_communi_bias=1111.0)
    dev_mem_cap = cost_model_context.get_cost_model_context("device_memory_capacity")
    assert math.isclose(dev_mem_cap, 32.0 * 1024.0 * 1024.0 * 1024.0, rel_tol=1e-6)
    costmodel_alpha = cost_model_context.get_cost_model_context("costmodel_alpha")
    assert math.isclose(costmodel_alpha, 1.0, rel_tol=1e-6)
    costmodel_beta = cost_model_context.get_cost_model_context("costmodel_beta")
    assert math.isclose(costmodel_beta, 60.0, rel_tol=1e-6)
    costmodel_gamma = cost_model_context.get_cost_model_context("costmodel_gamma")
    assert math.isclose(costmodel_gamma, 0.1, rel_tol=1e-6)
    costmodel_communi_threshold = cost_model_context.get_cost_model_context("costmodel_communi_threshold")
    assert math.isclose(costmodel_communi_threshold, 1024.0, rel_tol=1e-6)
    costmodel_communi_const = cost_model_context.get_cost_model_context("costmodel_communi_const")
    assert math.isclose(costmodel_communi_const, 2222.0, rel_tol=1e-6)
    costmodel_communi_bias = cost_model_context.get_cost_model_context("costmodel_communi_bias")
    assert math.isclose(costmodel_communi_bias, 1111.0, rel_tol=1e-6)

    cost_model_context.reset_cost_model_context()
    dev_mem_cap = cost_model_context.get_cost_model_context("device_memory_capacity")
    assert math.isclose(dev_mem_cap, 16.0 * 1024.0 * 1024.0 * 1024.0, rel_tol=1e-6)
    costmodel_alpha = cost_model_context.get_cost_model_context("costmodel_alpha")
    assert math.isclose(costmodel_alpha, 1.0, rel_tol=1e-6)
    costmodel_beta = cost_model_context.get_cost_model_context("costmodel_beta")
    assert math.isclose(costmodel_beta, 400.0, rel_tol=1e-6)
    costmodel_gamma = cost_model_context.get_cost_model_context("costmodel_gamma")
    assert math.isclose(costmodel_gamma, 0.001, rel_tol=1e-6)
    costmodel_communi_threshold = cost_model_context.get_cost_model_context("costmodel_communi_threshold")
    assert math.isclose(costmodel_communi_threshold, 2048.0, rel_tol=1e-6)
    costmodel_communi_const = cost_model_context.get_cost_model_context("costmodel_communi_const")
    assert math.isclose(costmodel_communi_const, 3072.0, rel_tol=1e-6)
    costmodel_communi_bias = cost_model_context.get_cost_model_context("costmodel_communi_bias")
    assert math.isclose(costmodel_communi_bias, 1024.0, rel_tol=1e-6)

    set_algo_parameters(tensor_slice_align_enable=False, tensor_slice_align_size=32,
                        fully_use_devices=False, elementwise_op_strategy_follow=False,
                        enable_algo_approxi=True, algo_approxi_epsilon=0.001)
    para_slice_align_enable = get_algo_parameters("tensor_slice_align_enable")
    assert not para_slice_align_enable
    para_slice_align_size = get_algo_parameters("tensor_slice_align_size")
    assert para_slice_align_size == 32
    fully_use_devices = get_algo_parameters("fully_use_devices")
    assert not fully_use_devices
    elementwise_op_strategy_follow = get_algo_parameters("elementwise_op_strategy_follow")
    assert not elementwise_op_strategy_follow
    enable_approxi = get_algo_parameters("enable_algo_approxi")
    assert enable_approxi
    algo_epsilon = get_algo_parameters("algo_approxi_epsilon")
    assert math.isclose(algo_epsilon, 0.001, rel_tol=1e-6)

    expecte_single_loop = False
    signle_loop = _get_algo_single_loop()
    assert expecte_single_loop == signle_loop
    expecte_single_loop = False
    _set_algo_single_loop(expecte_single_loop)
    signle_loop = _get_algo_single_loop()
    assert expecte_single_loop == signle_loop

    reset_algo_parameters()
    para_slice_align_enable = get_algo_parameters("tensor_slice_align_enable")
    assert not para_slice_align_enable
    para_slice_align_size = get_algo_parameters("tensor_slice_align_size")
    assert para_slice_align_size == 16
    fully_use_devices = get_algo_parameters("fully_use_devices")
    assert fully_use_devices
    elementwise_op_strategy_follow = get_algo_parameters("elementwise_op_strategy_follow")
    assert not elementwise_op_strategy_follow
    enable_approxi = get_algo_parameters("enable_algo_approxi")
    assert not enable_approxi
    algo_epsilon = get_algo_parameters("algo_approxi_epsilon")
    assert math.isclose(algo_epsilon, 0.1, rel_tol=1e-6)

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)

    net = NetWithLoss(Net())
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    reset_op_id()

    net.set_train()
    _cell_graph_executor.compile(net, x, y, b, phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search('MatMul-op', k) is not None:
            assert v == [[16, 1], [1, 1]]

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

import numpy as np
import mindspore.common.dtype as mstype
from mindspore import Tensor, context
from mindspore.nn import Cell
from mindspore.ops import operations as P

from parallel.utils.utils import ParallelValidator, compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

logits_ = Tensor(np.random.uniform(0, 1, [8, 8]), mstype.float32)
labels_ = Tensor(np.random.randint(0, 10, [8, 8]), mstype.float32)


class Net(Cell):
    def __init__(self, reduction, strategy=None):
        super(Net, self).__init__()
        self.kldiv_loss = P.KLDivLoss(reduction).shard(strategy)

    def construct(self, logits, labels):
        out = self.kldiv_loss(logits, labels)
        return out


def test_kldiv_loss_mean_auto_parallel():
    """
    Features: test KLDivLoss auto parallel
    Description: auto parallel, reduction is 'mean'
    Expectation: compile success
    """
    context.set_context(device_target="GPU")
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0, full_batch=True)
    reduction = 'mean'
    net = Net(reduction)
    compile_net(net, logits_, labels_)


def test_kldiv_loss_none_auto_parallel():
    """
    Features: test KLDivLoss auto parallel
    Description: auto parallel, reduction is 'none'
    Expectation: compile success
    """
    context.set_context(device_target="GPU")
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0, full_batch=True)
    reduction = 'none'
    net = Net(reduction)
    compile_net(net, logits_, labels_)


def test_kldiv_loss_sum_auto_parallel():
    """
    Features: test KLDivLoss auto parallel
    Description: auto parallel, reduction is 'sum'
    Expectation: compile success
    """
    context.set_context(device_target="GPU")
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0, full_batch=True)
    reduction = 'sum'
    net = Net(reduction)
    compile_net(net, logits_, labels_)


def test_kldiv_loss_mean_data_parallel():
    """
    Features: test KLDivLoss data parallel
    Description: data parallel, reduction is 'mean'
    Expectation: compile success
    """
    context.set_context(device_target="GPU")
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=1)
    reduction = 'mean'
    net = Net(reduction)
    phase = compile_net(net, logits_, labels_)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('AllReduce-0', ['KLDivLoss-0'])
    assert validator.check_node_attrs('AllReduce-0', {'op': 'sum'})


def test_kldiv_loss_none_data_parallel():
    """
    Features: test KLDivLoss data parallel
    Description: data parallel, reduction is 'none'
    Expectation: compile success
    """
    context.set_context(device_target="GPU")
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=1)
    reduction = 'none'
    net = Net(reduction)
    compile_net(net, logits_, labels_)


def test_kldiv_loss_none_model_parallel():
    """
    Features: test KLDivLoss model parallel
    Description: model parallel, reduction is 'none'
    Expectation: compile success
    """
    context.set_context(device_target="GPU")
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=5)
    reduction = 'none'
    strategy = ((2, 2), (2, 2))
    net = Net(reduction, strategy)
    compile_net(net, logits_, labels_)


def test_kldiv_loss_mean_model_parallel():
    """
    Features: test KLDivLoss model parallel
    Description: model parallel, reduction is 'mean'
    Expectation: compile success
    """
    context.set_context(device_target="GPU")
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=5)
    reduction = 'mean'
    strategy = ((4, 2), (4, 2))
    net = Net(reduction, strategy)
    phase = compile_net(net, logits_, labels_)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('AllReduce-0', ['KLDivLoss-0'])
    assert validator.check_node_attrs('AllReduce-0', {'op': 'sum'})

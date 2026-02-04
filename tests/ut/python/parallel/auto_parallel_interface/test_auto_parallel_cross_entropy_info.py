# Copyright 2025 Huawei Technologies Co., Ltd
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

import mindspore as ms
from mindspore.common.parameter import Parameter
from mindspore.graph.api import _cell_graph_executor
from mindspore.nn import Cell
from mindspore.ops import stop_gradient
from mindspore.ops import operations as P
from parallel.auto_parallel_interface._utils import init_hccl, set_parallel_mode
from parallel.auto_parallel_interface._utils_dataset import FakeData


ms.context.set_context(mode=ms.GRAPH_MODE)


class ParallelCrossEntropyNet(Cell):
    def __init__(self, mul_size, mul_strategy=None, cross_strategy=None, reduction="mean", ignore_index=-100,
                 label_smoothing=0.0):
        super().__init__()
        mul_np = np.full(mul_size, 0.5, dtype=np.float16)
        self.mul_weight = Parameter(ms.Tensor(mul_np, dtype=ms.float16), name="mul_weight")
        self.mul = P.Mul()
        self.cross_reduction = reduction
        self.cross_ignore_index = ignore_index
        self.cross_label_smoothing = label_smoothing
        self.cross_entropy = ms.ops.auto_generate.gen_ops_prim.CrossEntropyLoss()
        if cross_strategy is not None:
            self.mul.shard(mul_strategy)
            self.cross_entropy.shard(cross_strategy)

    def construct(self, inputs, label, cross_weight):
        x = self.mul(inputs, self.mul_weight)
        label = stop_gradient(label)
        loss = self.cross_entropy(x, label, cross_weight, self.cross_reduction, self.cross_ignore_index,
                                  self.cross_label_smoothing)
        return loss


def compile_graph(net, parallel_config, *inputs):
    net.set_train()
    net = set_parallel_mode(net, parallel_config)
    phase, _ = _cell_graph_executor.compile(net, *inputs)
    return phase


def test_cross_entropy_semi_auto_parallel_with_strategy():
    """
    Features: test CrossEntropyInfo in semi_auto_parallel mode
    Description: shard with tupe
    Expectation: compile success
    """
    init_hccl(global_rank=0, device_num=8)

    dataset = FakeData(size=256, batch_size=16, image_size=(96,), num_classes=96, use_parallel=True, data_num=3)
    dataset.set_label_onehot(is_onehot=False)
    inputs = dataset[0][0].astype(ms.float16)
    target = dataset[0][1].astype(ms.int64)
    weight = dataset[0][2].astype(ms.float16)

    # net
    net = ParallelCrossEntropyNet(mul_size=(128, 96), mul_strategy=((2, 1), (2, 1)),
                                  cross_strategy=((2, 1), (2,), (1,)))
    parallel_config = {"parallel_mode": "semi_auto"}

    # compile
    compile_graph(net, parallel_config, inputs, target, weight)


def test_cross_entropy_semi_auto_parallel_without_strategy():
    """
    Features: test CrossEntropyInfo in semi_auto parallel mode
    Description: shard without strategy
    Expectation: compile success
    """
    init_hccl(global_rank=0, device_num=8)

    dataset = FakeData(size=256, batch_size=16, image_size=(96,), num_classes=96, use_parallel=True, data_num=3)
    dataset.set_label_onehot(is_onehot=False)
    inputs = dataset[0][0].astype(ms.float16)
    target = dataset[0][1].astype(ms.int64)
    weight = dataset[0][2].astype(ms.float16)

    # net
    net = ParallelCrossEntropyNet(mul_size=(128, 96), mul_strategy=((2, 1), (2, 1)),
                                  cross_strategy=None)
    parallel_config = {"parallel_mode": "semi_auto"}

    # compile
    compile_graph(net, parallel_config, inputs, target, weight)

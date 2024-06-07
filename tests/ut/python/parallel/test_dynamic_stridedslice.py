# Copyright 2023 Huawei Technologies Co., Ltd
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
from mindspore import context, Tensor, Parameter
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from mindspore.ops import operations as P
from parallel.utils.utils import ParallelValidator


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, weight, strategy=None):
        super().__init__()
        self.strided_slice = P.StridedSlice()
        if strategy is not None:
            self.strided_slice.shard(strategy)
        self.weight = Parameter(weight, "w1")
        self.strides = (1, 1)
        self.gather = P.Gather()
        self.relu = P.ReLU()

    def construct(self, x, b):
        shape = P.Shape()(x)
        bs = shape[0]
        seq = shape[1]
        begin = (bs * 2 // 4, 0)
        end = (bs * 3 // 4, seq)
        out = self.strided_slice(x, begin, end, self.strides)
        out = self.gather(self.weight, out, 0)
        out = self.relu(out)
        return out


_x1 = Tensor(shape=[None, None], dtype=ms.int32)
_w1 = Tensor(np.ones([512, 8]), dtype=ms.float32)
_b1 = Tensor(shape=[None, None], dtype=ms.int32)


def compile_net(net, _x1, _b1):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    train_net.set_inputs(_x1, _b1)
    phase, _ = _cell_graph_executor.compile(train_net, _x1, _b1)
    context.reset_auto_parallel_context()
    return phase, train_net


def test_dynamic_stridedslice():
    """
    Features: StridedSlice dynamic
    Description:
    Expectation: No raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    net = Net(_w1)
    phase, _ = compile_net(net, _x1, _b1)
    validator = ParallelValidator(net, phase)

    # check inputs
    assert validator.check_node_inputs_has('Gather-0', ['StridedSlice-0'], graph_id=1)


def test_dynamic_stridedslice_handle_begin_end():
    """
    Features: StridedSlice dynamic
    Description:
    Expectation: No raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = (((1, 8),))
    net = Net(_w1, strategy)

    from mindspore import Symbol
    s1 = Symbol(divisor=1)
    s2 = Symbol(divisor=8)
    _x = Tensor(shape=[s1, s2], dtype=ms.int32)
    _b = Tensor(shape=[s1, s2], dtype=ms.int32)

    phase, _ = compile_net(net, _x, _b)
    validator = ParallelValidator(net, phase)

    # check inputs
    assert validator.check_node_inputs_has('Gather-0', ['Concat-0'], graph_id=1)

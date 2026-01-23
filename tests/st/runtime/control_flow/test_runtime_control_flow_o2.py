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

import numpy as np
import mindspore.ops.operations as op
from mindspore.nn import Cell
from mindspore import ops
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.dataset import GeneratorDataset
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.train.model import Model
from mindspore.nn.optim.momentum import Momentum
import mindspore.context as context
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})

def create_model(network, amp_level="O0", metrics=None, loss_scale_manager=None):
    loss = SoftmaxCrossEntropyWithLogits(reduction='mean')
    opt_fn = Momentum(learning_rate=0.01, momentum=0.9, params=network.get_parameters())
    model = Model(network=network, loss_fn=loss, optimizer=opt_fn, amp_level=amp_level, metrics=metrics,
                  loss_scale_manager=loss_scale_manager)
    return model

def generator_dataset(size, x_shape=(32, 1), y_shape=(32, 1)):
    for _ in range(size):
        x = np.full(x_shape, 0.1, dtype=np.float32)
        y = np.full(y_shape, 0.2, dtype=np.float32)
        yield(x, y)

def cond_func(init_value):
    return init_value[1] > 1

def print_while(init_value):
    input_tensor, init = init_value
    add = op.Add()
    print_ms = op.Print()
    out = add(input_tensor, init)
    print_ms("=========== test:", out)
    init -= 1
    return [out, init]

class WhileForPrint(Cell):
    def __init__(self, size_1=(1, 1)):
        super().__init__()
        self.add = op.Add()
        self.whileop = ops.WhileLoop()
        self.weight_1 = Parameter(Tensor(np.full(size_1, 0.5, dtype=np.float32)), name="weight_1")

    def construct(self, inputs, loop_times=3):
        res = self.whileop(cond_func, print_while, [inputs, loop_times])
        out = self.add(res[0], self.weight_1)
        return out

def load_model(model, epoch, dataset, dataset_sink_mode=True, asymc_save=False, sink_size=1,
               integrated_save=True, load_format="default"):
    model.train(epoch=epoch, train_dataset=dataset, dataset_sink_mode=dataset_sink_mode, sink_size=sink_size)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.0)
def test_compiler_while_op_o2():
    """
    Feature: Runtime tuple output to make tuple.
    Description: value tuple used more than once.
    Expectation: Not throw exception.
    """
    net = WhileForPrint()
    dataset = GeneratorDataset(lambda: generator_dataset(size=2), ["x", "y"])
    model = create_model(net)
    load_model(model, epoch=1, dataset=dataset, dataset_sink_mode=False)

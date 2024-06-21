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
import sys
import numpy as np
from mindspore.train import LossMonitor
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.train import Model
from mindspore.nn.wrap.cell_wrapper import PipelineCell, Cell
from mindspore import lazy_inline
from mindspore.communication import init
from mindspore.nn.optim import Momentum

ms.set_seed(1)

class GenDataset():
    def __init__(self, data_input, length=3):
        self.data = data_input
        self.index = 1
        self.length = length

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        self.index += 1
        return self.data

    def __iter__(self):
        return self

    @staticmethod
    def get_dataset_size():
        return 32

    @staticmethod
    def get_repeat_count():
        return 1

    @staticmethod
    def get_batch_size():
        return 32

    def create_tuple_iterator(self, num_epochs=1, do_copy=True):
        return self

    def reset(self):
        self.index = 0


class MatMulCell(Cell):
    def __init__(self):
        super().__init__()
        self.param = Parameter(initializer("zeros", [64, 64]), name="param")
        self.param1 = Parameter(initializer("zeros", [64, 64]), name="param1")
        self.matmul = P.MatMul()
        self.matmul1 = P.MatMul()

    def construct(self, x):
        out = self.matmul(x, self.param)
        out = self.matmul1(out, self.param1)
        return out


class Net(nn.Cell):
    @lazy_inline
    def __init__(self):
        super().__init__()
        self.block = nn.CellList()
        for i in range(8):
            cell = MatMulCell()
            cell.pipeline_stage = i
            self.block.append(cell)
        self.block[3].recompute()

    def construct(self, x):
        for i in range(8):
            x = self.block[i](x)
        return x


class LossCallBack(LossMonitor):
    def __init__(self, has_trained_epoch=0):
        super(LossCallBack, self).__init__()
        self.has_trained_epoch = has_trained_epoch

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss1 = cb_params.net_outputs

        if isinstance(loss1, (tuple, list)):
            if isinstance(loss1[0], Tensor) and isinstance(loss1[0].asnumpy(), np.ndarray):
                loss1 = loss1[0]

        if isinstance(loss1, Tensor) and isinstance(loss1.asnumpy(), np.ndarray):
            loss1 = np.mean(loss1.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss1, float) and (np.isnan(loss1) or np.isinf(loss1)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))
        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            print("epoch: %s step: %s, loss is %s" % (cb_params.cur_epoch_num + int(self.has_trained_epoch),
                                                      cur_step_in_epoch, loss1), flush=True)


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, enable_compile_cache=True, compile_cache_path=sys.argv[1])
    context.set_context(jit_level='O0')
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", pipeline_stages=8)
    init()
    data1 = Tensor(np.ones([32, 64]), dtype=ms.float32)
    dataset = GenDataset(data1, 3)
    net = PipelineCell(Net(), 8)
    learning_rate = 0.01
    momentum = 0.9
    optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate, momentum)
    model = Model(net, optimizer=optimizer)
    loss_cb = LossCallBack(0)
    cb = [loss_cb]
    model.train(2, dataset, dataset_sink_mode=False, callbacks=cb)
    context.set_context(enable_compile_cache=False)

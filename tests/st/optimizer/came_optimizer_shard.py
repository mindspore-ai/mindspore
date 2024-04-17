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
# ============================================================================
""" test came """
import os
import sys
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, Parameter, context, build_searched_strategy
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.ops import operations as P
from mindspore.train import Callback
from mindspore.train import Model
import mindspore.dataset as ds
from mindspore.communication import init
from came import Came

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
context.set_context(device_id=1)

class Net(nn.Cell):
    """ Net definition """
    def __init__(self):
        super(Net, self).__init__()
        self.weight = Parameter(Tensor(np.ones([64, 10]).astype(np.float32)), name="weight")
        self.bias = Parameter(Tensor(np.ones([10]).astype((np.float32))), name="bias")
        self.matmul = P.MatMul()
        self.biasAdd = P.BiasAdd()

    def construct(self, x):
        x = self.biasAdd(self.matmul(x, self.weight), self.bias)
        return x

class ModelCallback(Callback):
    def __init__(self):
        super(ModelCallback, self).__init__()
        self.loss_list = []

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        result = cb_params.net_outputs
        self.loss_list.append(result.asnumpy().mean())

class MyDataset:
    def __init__(self, n, in_dim, out_dim):
        self.input_data = []
        self.label_data = []
        for _ in range(n):
            self.input_data.append(np.arange(0.0, in_dim, dtype=np.float32) * 0.1)
            label_data = np.zeros(out_dim, dtype=np.float32)
            label_data[0] = 1.0
            self.label_data.append(label_data)

    def __getitem__(self, index):
        return self.input_data[index], self.label_data[index]

    def __len__(self):
        return len(self.input_data)

def came_compile():
    """ test came compile"""
    inputs = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = Net()
    net.set_train()

    loss = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = Came(net.trainable_params(), learning_rate=0.1)

    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _cell_graph_executor.compile(train_network, inputs, label)

def came_loss():
    """ test came with loss decrease"""
    net = Net()
    net.set_train()
    parallel_callback = ModelCallback()

    loss = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = Came(net.trainable_params(), learning_rate=0.1)
    fake_dataset = MyDataset(8, 64, 10)
    dataset = ds.GeneratorDataset(fake_dataset, ["input", "label"]).batch(1)
    model = Model(net, loss_fn=loss, optimizer=optimizer)
    model.train(1, dataset, dataset_sink_mode=False, callbacks=parallel_callback)
    loss_values = np.array(parallel_callback.loss_list)
    assert abs(loss_values[-1]) < abs(loss_values[0])

def came_parallel():
    "test came optimizer shard with two cards with loss decrease"
    strategy_file_path = "./strategy_stage1.ckpt"
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
    ms.set_auto_parallel_context(enable_parallel_optimizer=True)
    ms.set_auto_parallel_context(strategy_ckpt_config={"save_file": strategy_file_path})
    ms.set_auto_parallel_context(parallel_optimizer_config={"parallel_optimizer_threshold": 1})
    context.set_context(device_id=int(os.getenv('DEVICE_ID')))
    init()
    net = Net()
    net.set_train()
    parallel_callback = ModelCallback()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = Came(net.trainable_params(), learning_rate=0.1)
    fake_dataset = MyDataset(8, 64, 10)
    dataset = ds.GeneratorDataset(fake_dataset, ["input", "label"]).batch(2)
    model = Model(net, loss_fn=loss, optimizer=optimizer)
    model.train(1, dataset, dataset_sink_mode=False, callbacks=parallel_callback)
    assert optimizer.exp_avg_sq_row[0].shape == (32,)
    assert optimizer.exp_avg_sq_row[1].shape == (1,)
    assert optimizer.exp_avg_sq_col[0].shape == (10,)
    assert optimizer.exp_avg_sq_col[1].shape == (1,)
    assert optimizer.exp_avg_insta_row[0].shape == (32,)
    assert optimizer.exp_avg_insta_row[1].shape == (1,)
    assert optimizer.exp_avg_insta_col[0].shape == (10,)
    assert optimizer.exp_avg_insta_col[1].shape == (1,)
    assert optimizer.exp_avg_sq[0].shape == (1,)
    assert optimizer.exp_avg_sq[1].shape == (10,)
    loss_values = np.array(parallel_callback.loss_list)
    assert abs(loss_values[-1]) < abs(loss_values[0])
    strategy = build_searched_strategy(strategy_file_path)
    matched_count = 0
    for key, _ in strategy.items():
        if 'avg' in key:
            matched_count += 1
    assert matched_count == 7

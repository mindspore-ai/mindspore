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
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, context, ms_function
from mindspore import dataset as ds
from mindspore.ops import operations as P
from mindspore.common.jit_config import JitConfig


context.set_context(mode=ms.GRAPH_MODE)
context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", dataset_strategy="full_batch")


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Attention(nn.Cell):
    def __init__(self):
        super(Attention, self).__init__()
        self.fc_a = nn.Dense(128, 768, activation='relu')
        self.fc_b = nn.Dense(128, 768, activation='relu')
        self.fc_c = nn.Dense(128, 768, activation='relu')
        self.fc_a.matmul.shard(((1, 1), (8, 1)))
        self.fc_b.matmul.shard(((1, 1), (8, 1)))
        self.fc_c.matmul.shard(((1, 1), (8, 1)))

    def construct(self, x):
        q = self.fc_a(x)
        k = self.fc_b(x)
        v = self.fc_c(x)
        return q, k, v

attention = Attention()
relu = nn.ReLU()


@ms_function
def dense_func(x, label):
    q, k, v = attention(x)
    k = P.Transpose()(k, (1, 0)) # (728, 32)
    c = relu(P.MatMul()(q, k)) # (32, 32)
    s = relu(P.MatMul()(c, v)) # (32, 768)
    s = s - label
    return P.ReduceMean()(s * s)

optimizer_adam = nn.Adam(attention.trainable_params(), learning_rate=0.001)
attention.set_train()
attention.update_parameters_name("attn")
optimizer_adam.update_parameters_name("opt")
grad_dens_func = ms.ops.value_and_grad(dense_func, None, optimizer_adam.parameters)


@ms_function
def train_step(input_, label_):
    loss, grad = grad_dens_func(input_, label_)
    optimizer_adam(grad)
    return loss


def test_sink():
    """
    Feature: Function mode in auto parallel
    Description: sink mode
    Expectation: compile ok
    """
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel",
                                      dataset_strategy="data_parallel", device_num=8)
    data = {"input": np.ones([16, 32, 128]).astype(np.float32), "label": np.zeros([16, 32, 768]).astype(np.float32)}
    dataset = ds.NumpySlicesDataset(data=data)
    jitconfig = JitConfig(jit_level="O1", task_sink=True)
    sink_process = ms.train.data_sink(dense_func, dataset, sink_size=4, jit_config=jitconfig)
    _ = sink_process()


def test_no_sink():
    """
    Feature: Function mode in auto parallel
    Description: no sink mode
    Expectation: compile ok
    """
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", dataset_strategy="full_batch", device_num=8)
    _ = dense_func(Tensor(np.ones([32, 128]).astype(np.float32)), Tensor(np.zeros([32, 768]).astype(np.float32)))


def test_sink_with_grad():
    """
    Feature: Function mode in auto parallel
    Description: sink mode with grad
    Expectation: compile ok
    """
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel",
                                      dataset_strategy="data_parallel", device_num=8)
    data = {"input": np.ones([16, 32, 128]).astype(np.float32), "label": np.zeros([16, 32, 768]).astype(np.float32)}
    dataset = ds.NumpySlicesDataset(data=data)
    jitconfig = JitConfig(jit_level="O1", task_sink=True)
    sink_process = ms.train.data_sink(train_step, dataset, sink_size=4, jit_config=jitconfig)
    _ = sink_process()


def test_no_sink_with_grad():
    """
    Feature: Function mode in auto parallel
    Description: no sink mode with grad
    Expectation: compile ok
    """
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", dataset_strategy="full_batch", device_num=8)
    _ = train_step(Tensor(np.ones([32, 128]).astype(np.float32)), Tensor(np.zeros([32, 768]).astype(np.float32)))

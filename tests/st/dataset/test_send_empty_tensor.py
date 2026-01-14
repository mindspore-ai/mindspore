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
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import Callback, ops
from mindspore.train import Model
from tests.mark_utils import arg_mark


class MyDataset:
    def __init__(self):
        self.data = [np.array([]), [[]]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class Net(nn.Cell):
    def construct(self, x):
        return ops.square(x)


class SaveLossCallback(Callback):
    def __init__(self):
        super(SaveLossCallback, self).__init__()
        self.loss = []

    def step_end(self, run_context):
        loss = run_context.original_args().net_outputs
        self.loss.append(loss.asnumpy())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_dataset_sink_mode_send_empty_tensor():
    """
    Feature: Dataset iterator does not validate empty tensor any more.
    Description: Dataset iterator returns an empty array to model, and then model can compute successfully.
    Expectation: Model can compute successfully.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    net = Net()
    model = Model(net)
    dataset = ds.GeneratorDataset(MyDataset(), ["A"], shuffle=False)

    loss_callback = SaveLossCallback()
    model.train(1, dataset, dataset_sink_mode=True, sink_size=-1)
    np.testing.assert_array_equal(loss_callback.loss, [])

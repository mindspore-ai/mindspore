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

import numpy as np

import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import Callback, ops
from mindspore.train import Model
from tests.mark_utils import arg_mark


class MyDataset:
    def __init__(self):
        self.data = [np.array(1), np.array(2), np.array(3), np.array(4)]

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
def test_dataset_iterator_mode_create_tensor_without_copy():
    """
    Feature: Dataset iterator creates a Tensor from numpy.ndarray without copy.
    Description: Compare result of 'do_copy=True' and 'do_copy=False'.
    Expectation: Expected two results are the same.
    """
    net = Net()
    model = Model(net)

    dataset = ds.GeneratorDataset(MyDataset(), ["A"], shuffle=False)
    dataset = dataset.batch(2)
    epoch = 1

    # do_copy works in non dataset sink mode
    ds.config.set_iterator_mode(do_copy=False, parallel_convert=False)
    loss_callback1 = SaveLossCallback()
    model.train(epoch, dataset, callbacks=loss_callback1, dataset_sink_mode=False)
    copy_false_loss = loss_callback1.loss

    ds.config.set_iterator_mode(do_copy=True, parallel_convert=False)
    loss_callback2 = SaveLossCallback()
    model.train(epoch, dataset, callbacks=loss_callback2, dataset_sink_mode=False)
    copy_true_loss = loss_callback2.loss

    assert len(copy_true_loss) == len(copy_false_loss)
    for loss_false, loss_true in zip(copy_false_loss, copy_true_loss):
        np.testing.assert_array_equal(loss_false, loss_true)

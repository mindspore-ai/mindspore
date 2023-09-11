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
import pytest

import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import Callback, context, ops
from mindspore.common import set_seed
from mindspore.train import Model

set_seed(1)


def create_np_dataset(size, num_parallel_workers, python_multiprocessing):
    """
    Create a simple dataset.
    """
    data = np.array(list(range(size))).astype(np.int32)
    dataset = ds.NumpySlicesDataset(data, shuffle=True)

    def process(number):
        return number + 1

    dataset = dataset.map(operations=process, num_parallel_workers=num_parallel_workers,
                          python_multiprocessing=python_multiprocessing)
    return dataset


def create_model():
    """
    Create a simple model.
    """

    class Net(nn.Cell):
        def construct(self, x):
            return ops.square(x)

    net = Net()
    model = Model(net)

    return model


class SaveLossCallback(Callback):
    def __init__(self):
        super(SaveLossCallback, self).__init__()
        self.loss = []

    def step_end(self, run_context):
        loss = run_context.original_args().net_outputs
        self.loss.append(loss.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("sink_mode", (False, True))
@pytest.mark.parametrize("num_parallel_workers", (1, 4))
@pytest.mark.parametrize("python_multiprocessing", (False, True))
def test_set_init_step(sink_mode, num_parallel_workers, python_multiprocessing):
    """
    Feature: Pipeline resuming
    Description: Test resuming training in model.train
    Expectation: Model can resume training at the given step point
    """
    context.set_context(mode=context.GRAPH_MODE)
    original_seed = ds.config.get_seed()
    ds.config.set_seed(0)

    dataset_size = 20
    dataset = create_np_dataset(dataset_size, num_parallel_workers, python_multiprocessing)
    model = create_model()
    num_epochs = 3

    # train the whole dataset and save the expected loss of each step
    loss_callback = SaveLossCallback()
    model.train(num_epochs, dataset, callbacks=loss_callback, dataset_sink_mode=sink_mode)
    expected_loss = np.array(loss_callback.loss)

    # set initial step to the end of the first epoch and verify the loss
    init_step = 20
    dataset.set_init_step(init_step)
    loss_callback = SaveLossCallback()
    model.train(num_epochs, dataset, callbacks=loss_callback, dataset_sink_mode=sink_mode, initial_epoch=1)

    if not sink_mode:  # we have loss for each step
        skip_loss = init_step
    else:  # we only have loss for each epoch
        skip_loss = init_step // dataset_size
    for index, loss in enumerate(loss_callback.loss):
        np.testing.assert_array_equal(np.array(loss), expected_loss[index + skip_loss])

    ds.config.set_seed(original_seed)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("sink_mode", (False, True))
def test_set_init_step_at_intermediate_step(sink_mode):
    """
    Feature: Pipeline resuming
    Description: Test setting the resuming step to the intermediate step of epoch
    Expectation: The resuming point can be automatically reset to the end of the previous epoch in sink mode.
    """
    context.set_context(mode=context.GRAPH_MODE)
    original_seed = ds.config.get_seed()
    ds.config.set_seed(0)

    dataset_size = 20
    dataset = create_np_dataset(dataset_size, 1, False)
    model = create_model()
    num_epochs = 3

    # train the whole dataset and save the expected loss of each step
    loss_callback = SaveLossCallback()
    model.train(num_epochs, dataset, callbacks=loss_callback, dataset_sink_mode=sink_mode)
    expected_loss = np.array(loss_callback.loss)

    # set initial step to the intermediate of the second epoch at step 30,
    # then it will be automatically reset to the end of the first epoch at step 20
    init_step = 30
    dataset.set_init_step(init_step)
    loss_callback = SaveLossCallback()
    model.train(num_epochs, dataset, callbacks=loss_callback, dataset_sink_mode=sink_mode, initial_epoch=1)

    if not sink_mode:
        skip_loss = init_step
    else:
        actual_init_step = init_step // dataset_size * dataset_size
        skip_loss = actual_init_step // dataset_size
    for index, loss in enumerate(loss_callback.loss):
        np.testing.assert_array_equal(np.array(loss), expected_loss[index + skip_loss])

    ds.config.set_seed(original_seed)


if __name__ == '__main__':
    test_set_init_step(sink_mode=False, num_parallel_workers=4,
                       python_multiprocessing=True)
    test_set_init_step_at_intermediate_step(sink_mode=True)

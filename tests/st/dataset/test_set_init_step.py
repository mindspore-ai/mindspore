# Copyright 2023-2025 Huawei Technologies Co., Ltd
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
"""test dataset set_init_step"""
import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore import nn
from mindspore import Callback, context, ops
from mindspore.train import Model
from tests.mark_utils import arg_mark


class SaveLossCallback(Callback):
    """
    Callback to save loss of each step.
    """

    def __init__(self):
        super().__init__()
        self.loss = []

    def on_train_step_end(self, run_context):
        """
        Save losses into a list at each step end.
        """
        loss = run_context.original_args().net_outputs
        self.loss.append(loss.asnumpy())


class Net(nn.Cell):
    """
    A very simple network.
    """

    def construct(self, x):
        """
        Take the square of the input.
        """
        return ops.square(x)


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown_seed():
    """
    Set seed before the testcase to keep the randomness of shuffle then reset it.
    """
    original_seed = ds.config.get_seed()
    ds.config.set_seed(0)

    yield

    ds.config.set_seed(original_seed)


def create_np_dataset(dataset_size, num_parallel_workers, python_multiprocessing, distribute_sampler=None):
    """
    Create a simple dataset.
    """
    array = np.array(list(range(dataset_size))).astype(np.int32)
    if distribute_sampler is None:
        dataset = ds.NumpySlicesDataset(array, shuffle=True)
    else:
        dataset = ds.NumpySlicesDataset(array, sampler=distribute_sampler)

    def process(data):
        """
        A simple python function to add input data by 1.
        """
        return data + 1

    dataset = dataset.map(operations=process, num_parallel_workers=num_parallel_workers,
                          python_multiprocessing=python_multiprocessing)
    return dataset


def create_model():
    """
    Create a simple model.
    """
    net = Net()
    model = Model(net)
    return model


def train_and_get_loss(dataset, model, num_epochs, sink_mode, sink_size, initial_epoch=0):
    """
    Train the model and return the loss.
    """
    loss_callback = SaveLossCallback()
    model.train(num_epochs, dataset, callbacks=loss_callback, dataset_sink_mode=sink_mode,
                sink_size=sink_size, initial_epoch=initial_epoch)
    loss = np.array(loss_callback.loss)
    return loss


def validate_retrain_loss_equal_to_normal_train(mode, backend, sink_mode, dataset_size, init_step, sink_size=-1,
                                                distribute_sampler=None):
    """
    Verify that the result of breakpoint training in each scenario is the same as normal training.
    """
    context.set_context(mode=mode)

    if sink_mode and sink_size != -1:
        num_steps_per_epoch = sink_size
    else:
        num_steps_per_epoch = dataset_size

    if backend != "CPU":
        # test both multiprocessing and multi-threading
        multiprocess_cases = [False, True]
    else:
        # our CPU CI machines have poor memory that can not afford multiprocessing
        multiprocess_cases = [False]
    for python_multiprocessing in multiprocess_cases:
        num_parallel_workers = 1
        # create a simple data pipeline and model without randomness
        dataset = create_np_dataset(dataset_size, num_parallel_workers, python_multiprocessing, distribute_sampler)
        model = create_model()

        # train the whole dataset and save the expected loss of each step
        num_epochs = 2
        expected_loss = train_and_get_loss(dataset, model, num_epochs, sink_mode, sink_size)

        # set initial step and verify the loss
        dataset.set_init_step(init_step)
        init_epoch = init_step // num_steps_per_epoch
        retrain_loss = train_and_get_loss(dataset, model, num_epochs, sink_mode, sink_size,
                                          initial_epoch=init_epoch)

        if sink_mode and backend != "CPU":
            if mode == context.GRAPH_MODE:
                # we only have loss for each epoch
                skip_loss_count = init_step // num_steps_per_epoch
            else:
                # we have loss for each step, but the init_step will be
                # set to the epoch end due to sink mode
                skip_loss_count = init_step // num_steps_per_epoch * num_steps_per_epoch
        else:
            # we have loss for each step
            skip_loss_count = init_step

        assert len(expected_loss) == len(retrain_loss) + skip_loss_count
        for index, loss in enumerate(retrain_loss):
            np.testing.assert_array_equal(np.array(loss), expected_loss[index + skip_loss_count])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", (context.GRAPH_MODE, context.PYNATIVE_MODE))
@pytest.mark.parametrize("sink_mode", (False, True))
def test_set_init_step_cpu(mode, sink_mode):
    """
    Feature: Pipeline resuming
    Description: Test resuming training by model.train on CPU
    Expectation: Model can resume training at the given step point
    """
    # set initial step to the end of the first epoch and verify the loss
    init_step = 10
    dataset_size = 10
    validate_retrain_loss_equal_to_normal_train(mode, "CPU", sink_mode, dataset_size, init_step)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", (context.GRAPH_MODE, context.PYNATIVE_MODE))
@pytest.mark.parametrize("sink_mode", (False, True))
def test_set_init_step_cpu_with_distribute_sampler(mode, sink_mode):
    """
    Feature: Pipeline resuming
    Description: Test resuming training by model.train on CPU
    Expectation: Model can resume training at the given step point
    """
    # set initial step to the end of the first epoch and verify the loss
    init_step = 4
    dataset_size = 80
    distribute_sampler = ds.DistributedSampler(num_shards=8, shard_id=4)
    validate_retrain_loss_equal_to_normal_train(mode, "CPU", sink_mode, dataset_size, init_step, -1, distribute_sampler)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize("mode", (context.GRAPH_MODE, context.PYNATIVE_MODE))
@pytest.mark.parametrize("sink_mode_and_size", ((False, -1), (True, -1), (True, 10)))
def test_set_init_step_device(mode, sink_mode_and_size: tuple):
    """
    Feature: Pipeline resuming
    Description: Test resuming training by model.train on GPU and Ascend
    Expectation: Model can resume training at the given step point
    """
    # set initial step to the end of the first epoch and verify the loss
    init_step = 10
    dataset_size = 10
    sink_mode, sink_size = sink_mode_and_size
    validate_retrain_loss_equal_to_normal_train(mode, "DEVICE", sink_mode, dataset_size, init_step, sink_size)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", (context.GRAPH_MODE, context.PYNATIVE_MODE))
@pytest.mark.parametrize("sink_mode", (False, True))
def test_set_init_step_at_intermediate_step_cpu(mode, sink_mode):
    """
    Feature: Pipeline resuming
    Description: Test setting the resuming step to the intermediate step of epoch on CPU
    Expectation: The resuming point can be automatically reset to the end of the previous epoch in sink mode
    """
    # set initial step to the intermediate of the second epoch at step 15,
    # then it will be automatically reset to the end of the first epoch at step 10
    init_step = 15
    dataset_size = 10
    validate_retrain_loss_equal_to_normal_train(mode, "CPU", sink_mode, dataset_size, init_step)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize("mode", (context.GRAPH_MODE, context.PYNATIVE_MODE))
@pytest.mark.parametrize("sink_mode_and_size", ((False, -1), (True, -1), (True, 10)))
def test_set_init_step_at_intermediate_step_device(mode, sink_mode_and_size: tuple):
    """
    Feature: Pipeline resuming
    Description: Test setting the resuming step to the intermediate step of epoch on GPU and Ascend
    Expectation: The resuming point can be automatically reset to the end of the previous epoch in sink mode
    """
    # set initial step to the intermediate of the second epoch at step 12,
    # then it will be automatically reset to the end of the first epoch at step 10
    init_step = 12
    dataset_size = 10
    sink_mode, sink_size = sink_mode_and_size
    validate_retrain_loss_equal_to_normal_train(mode, "DEVICE", sink_mode, dataset_size, init_step, sink_size)


if __name__ == '__main__':
    test_set_init_step_cpu(mode=context.GRAPH_MODE, sink_mode=True)
    test_set_init_step_device(mode=context.GRAPH_MODE, sink_mode_and_size=(True, -1))
    test_set_init_step_at_intermediate_step_cpu(mode=context.GRAPH_MODE, sink_mode=True)
    test_set_init_step_at_intermediate_step_device(mode=context.GRAPH_MODE, sink_mode_and_size=(True, -1))

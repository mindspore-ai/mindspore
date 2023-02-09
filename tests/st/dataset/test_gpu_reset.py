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
# ============================================================================
import pytest

import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import context
from mindspore import log as logger
from mindspore.common import set_seed
from mindspore.ops import operations as P
from mindspore.parallel._recovery_context import _get_recovery_context_func_map
from mindspore.train import Model, Callback

set_seed(1)

# pylint: disable=no-value-for-parameter


def create_np_dataset(size, num_parallel_workers, python_multiprocessing):
    """
    Create dataset for train or test
    """
    def my_func(x):
        return x + 1 if x % 2 else x - 1
    data = ds.NumpySlicesDataset([(x,) for x in range(1, size + 1)], shuffle=False)
    data = data.map(operations=my_func, num_parallel_workers=num_parallel_workers,
                    python_multiprocessing=python_multiprocessing)
    return data


def create_model():
    """
    Define and return a simple model
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.print = P.Print()

        def construct(self, x):
            self.print(x)
            return x

    net = Net()
    model_ = Model(net)

    return model_


class MyCallback(Callback):
    def __init__(self, dataset_size, reset_point):
        self.dataset_size = dataset_size
        self.reset_point = reset_point

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        logger.info(f"Epoch #{cb_params.cur_epoch_num - 1} has ended")
        if cb_params.cur_epoch_num == self.reset_point:
            dataset = ds.engine.datasets._get_training_dataset()  # pylint: disable=W0212
            dataset._reset(self.reset_point * self.dataset_size, self.reset_point)  # pylint: disable=W0212


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("fast_recovery", (False, True))
@pytest.mark.parametrize("num_parallel_workers", (1, 4))
@pytest.mark.parametrize("python_multiprocessing", (False, True))
@pytest.mark.forked
def test_dataset_reset_sink(fast_recovery, num_parallel_workers, python_multiprocessing):
    """
    Feature: Dataset recovery
    Description: Test Dataset recovery when GPU (and sink mode) is used.
    Expectation: Training completes successfully
    """
    def enable_recovery():
        """Get whether enable recovery"""
        return True

    context.set_context(mode=context.GRAPH_MODE)
    _get_recovery_context_func_map["enable_recovery"] = enable_recovery
    original_fast_recovery = ds.config.get_fast_recovery()
    ds.config.set_fast_recovery(fast_recovery)
    data = create_np_dataset(20, num_parallel_workers, python_multiprocessing)
    model = create_model()
    num_epochs = 3
    reset_point = 2  # 2nd epoch
    cb = MyCallback(dataset_size=data.get_dataset_size(), reset_point=reset_point)
    model.train(num_epochs, data, callbacks=[cb], dataset_sink_mode=True)
    ds.config.set_fast_recovery(original_fast_recovery)


if __name__ == '__main__':
    test_dataset_reset_sink(fast_recovery=True, num_parallel_workers=4, python_multiprocessing=True)

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

"""Dataset regression coverage for gpu reset."""
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import log as logger
from mindspore.common import set_seed
from mindspore.ops import operations as P
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
            super().__init__()
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
            dataset._reset(self.reset_point * self.dataset_size, self.dataset_size)  # pylint: disable=W0212

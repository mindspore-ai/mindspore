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
import os
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore.train import Model
import mindspore.dataset as ds
from mindspore import log as logger
from mindspore.common import Tensor

# pylint: disable=no-value-for-parameter


def create_dataset(size, needs_batch):
    """
    Create dataset for train or test
    """
    def my_func(x):
        arr = np.zeros((2, 2))
        return ({"originally_tensor": x, "originally_numpy": arr, "originally_dict": {"dd": x},
                 "originally_int": 1, "originally_bool": True, "originally_float": 1.0}, x, arr)

    data_path = os.path.join("/home/workspace/mindspore_dataset/mnist", "train")
    data = ds.MnistDataset(data_path, num_parallel_workers=8, num_samples=size)
    data = data.project("image")
    data = data.map(operations=my_func, input_columns=["image"],
                    output_columns=["dict", "originally_tensor", "originally_numpy"])
    if needs_batch:
        data = data.batch(2)
    return data


def create_model():
    """
    Define and return a simple model
    """

    class Net(nn.Cell):
        def construct(self, x, y, z):
            assert isinstance(x, dict)
            assert isinstance(y, Tensor)
            assert isinstance(z, Tensor)
            return x

    net = Net()
    model_ = Model(net)

    return model_


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.forked
@pytest.mark.parametrize("needs_batch", (False, True))
def test_python_dict_in_pipeline(needs_batch):
    """
    Feature: Dataset pipeline contains a Python dict object
    Description: A dict object is created and sent to the model by dataset pipeline
    Expectation: Python dict object is successfully sent to the model
    """
    logger.info("test_python_dict_in_pipeline - dict object testing")

    num_epochs = 2
    dataset_size = 50
    data = create_dataset(dataset_size, needs_batch)
    model = create_model()

    # non-sink mode supports python dictionary
    model.train(num_epochs, data, dataset_sink_mode=False)

    # sink mode doesn't support python dict as input
    with pytest.raises(RuntimeError) as error_info:
        model.train(num_epochs, data, dataset_sink_mode=True)
    assert "The python type <class 'numpy.object_'> cannot be converted to MindSpore type." in str(
        error_info.value)


if __name__ == '__main__':
    test_python_dict_in_pipeline(True)

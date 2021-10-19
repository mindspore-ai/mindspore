# Copyright 2020 Huawei Technologies Co., Ltd
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
"""test dataset helper."""

import pytest
import numpy as np
import mindspore.context as context
from mindspore.communication.management import init
from mindspore.train.dataset_helper import DatasetHelper
from mindspore.communication._comm_helper import GlobalComm
from ....dataset_mock import MindData

def get_dataset(batch_size=1):
    dataset_types = (np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32)
    dataset_shapes = ((batch_size, 128), (batch_size, 128), (batch_size, 128), (batch_size, 1),
                      (batch_size, 20), (batch_size, 20), (batch_size, 20))

    dataset = MindData(size=2, batch_size=batch_size, np_types=dataset_types,
                       output_shapes=dataset_shapes, input_indexs=(0, 1))
    return dataset


def test_dataset_helper_dataset_sink_mode_str():
    dataset = get_dataset(32)
    with pytest.raises(TypeError):
        DatasetHelper(dataset, dataset_sink_mode="True")


def test_dataset_helper_dataset_sink_mode_int():
    dataset = get_dataset(32)
    with pytest.raises(TypeError):
        DatasetHelper(dataset, dataset_sink_mode=1)


def test_dataset_helper_sink_size_bool():
    dataset = get_dataset(32)
    with pytest.raises(TypeError):
        DatasetHelper(dataset, dataset_sink_mode=True, sink_size=True)


def test_dataset_helper_sink_size_float():
    dataset = get_dataset(32)
    with pytest.raises(TypeError):
        DatasetHelper(dataset, dataset_sink_mode=True, sink_size=1.0)


def test_dataset_helper_sink_size_negative():
    dataset = get_dataset(32)
    with pytest.raises(ValueError):
        DatasetHelper(dataset, dataset_sink_mode=True, sink_size=-2)


def test_dataset_iter_normal():
    dataset = get_dataset(32)
    dataset_helper = DatasetHelper(dataset, dataset_sink_mode=False)
    count = 0
    for _ in range(2):
        for _ in dataset_helper:
            count += 1
        dataset.reset()
    assert count == 6


@pytest.mark.skipif('not context.get_context("enable_ge")')
def test_dataset_iter_ge():
    GlobalComm.CHECK_ENVS = False
    init("hccl")
    GlobalComm.CHECK_ENVS = True
    dataset = get_dataset(32)
    dataset_helper = DatasetHelper(dataset, dataset_sink_mode=True, sink_size=10)
    count = 0
    for _ in range(2):
        for _ in dataset_helper:
            count += 1
    assert count == 2


@pytest.mark.skipif('context.get_context("enable_ge")')
def test_dataset_iter_ms():
    context.set_context(device_target='Ascend', mode=context.GRAPH_MODE)
    GlobalComm.CHECK_ENVS = False
    init("hccl")
    GlobalComm.CHECK_ENVS = True
    dataset = get_dataset(32)
    DatasetHelper(dataset, dataset_sink_mode=True, sink_size=10)

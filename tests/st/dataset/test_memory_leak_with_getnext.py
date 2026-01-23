# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
Test Map op on windows
"""

import os
import psutil
import time
import numpy as np

import mindspore.dataset as ds
from mindspore import Tensor, context
from mindspore.ops import operations as P
from mindspore.graph.api import _cell_graph_executor

from tests.mark_utils import arg_mark


class StaticData:
    """data with static shape"""
    def __init__(self, dataset_size):
        self.dataset_size = dataset_size
        self.input_ids = np.ones((1024, 1024), dtype=np.int32)  # 4M
        self.input_mask = np.ones((1024, 1024), dtype=np.int32)  # 4M
        self.segment_ids = np.ones((1024, 1024), dtype=np.int32)  # 4M
        self.next_sentence_labels = np.ones((1024, 1024), dtype=np.int32)  # 4M
        self.masked_lm_positions = np.ones((1024, 1024), dtype=np.int32)  # 4M
        self.masked_lm_ids = np.ones((1024, 1024), dtype=np.int32)  # 4M
        self.masked_lm_weights = np.ones((1024, 1024), dtype=np.float32)  # 4M

    def __getitem__(self, index):
        return self.input_ids, self.input_mask, self.segment_ids, self.next_sentence_labels, \
            self.masked_lm_positions, self.masked_lm_ids, self.masked_lm_weights

    def __len__(self):
        return self.dataset_size


class DynamicData:
    """data with dynamic shape"""
    def __init__(self, dataset_size):
        self.dataset_size = dataset_size
        self.input_ids = [np.ones((1024, 1024), dtype=np.int32), np.ones((768, 1024), dtype=np.int32),
                          np.ones((512, 1024), dtype=np.int32), np.ones((128, 1024), dtype=np.int32)]
        self.input_mask = [np.ones((1024, 1024), dtype=np.int32), np.ones((768, 1024), dtype=np.int32),
                           np.ones((512, 1024), dtype=np.int32), np.ones((128, 1024), dtype=np.int32)]
        self.segment_ids = [np.ones((1024, 1024), dtype=np.int32), np.ones((768, 1024), dtype=np.int32),
                            np.ones((512, 1024), dtype=np.int32), np.ones((128, 1024), dtype=np.int32)]
        self.next_sentence_labels = [np.ones((1024, 1024), dtype=np.int32), np.ones((768, 1024), dtype=np.int32),
                                     np.ones((512, 1024), dtype=np.int32), np.ones((128, 1024), dtype=np.int32)]
        self.masked_lm_positions = [np.ones((1024, 1024), dtype=np.int32), np.ones((768, 1024), dtype=np.int32),
                                    np.ones((512, 1024), dtype=np.int32), np.ones((128, 1024), dtype=np.int32)]
        self.masked_lm_ids = [np.ones((1024, 1024), dtype=np.int32), np.ones((768, 1024), dtype=np.int32),
                              np.ones((512, 1024), dtype=np.int32), np.ones((128, 1024), dtype=np.int32)]
        self.masked_lm_weights = [np.ones((1024, 1024), dtype=np.int32), np.ones((768, 1024), dtype=np.int32),
                                  np.ones((512, 1024), dtype=np.int32), np.ones((128, 1024), dtype=np.int32)]

    def __getitem__(self, index):
        return self.input_ids[index % 4], self.input_mask[index % 4], self.segment_ids[index % 4], \
            self.next_sentence_labels[index % 4], self.masked_lm_positions[index % 4], self.masked_lm_ids[index % 4], \
            self.masked_lm_weights[index % 4]

    def __len__(self):
        return self.dataset_size


def convert_type(shapes, types):
    """convert data type"""
    ms_types = []
    for np_shape, np_type in zip(shapes, types):
        input_np = np.zeros(np_shape, np_type)
        tensor = Tensor(input_np)
        ms_types.append(tensor.dtype)
    return ms_types


def get_dataset_shapes_and_types(dataset):
    """"get shape and types"""
    dataset_shapes = dataset.output_shapes()
    np_types = dataset.output_types()
    dataset_types = convert_type(dataset_shapes, np_types)
    return dataset_shapes, dataset_types


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=30.00)
def test_generatordataset_static_shape_memory():
    """
    Feature: Test memory leak with getnext when static shape
    Description: Testing static shape
    Expectation: Success
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", device_id=6, pynative_synchronize=True)
    dataset_size = 10000
    num_epochs = 8
    dataset = ds.GeneratorDataset(StaticData(dataset_size),
                                  ["input_ids", "input_mask", "segment_ids", "next_sentence_labels",
                                   "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"], num_parallel_workers=1)

    dataset_shapes, dataset_types = get_dataset_shapes_and_types(dataset)
    batch_size = dataset.get_batch_size()
    dataset = dataset.device_que()
    queue_name = dataset.queue_name
    _cell_graph_executor.init_dataset(queue_name, 1, batch_size, dataset_types, dataset_shapes, (), "")
    time.sleep(1)
    dataset.send(num_epochs)
    get_next = P.GetNext(dataset_types, dataset_shapes, len(dataset_types), queue_name)
    start_memory = psutil.Process(os.getpid()).memory_info().rss
    for _ in range(num_epochs * dataset_size):
        get_next()
    end_memory = psutil.Process(os.getpid()).memory_info().rss
    assert end_memory < start_memory + 2147483648


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=30.00)
def test_generatordataset_dynamic_shape_memory():
    """
    Feature: Test memory leak with getnext when dynamic shape
    Description: Testing dynamic shape
    Expectation: Success
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", device_id=6, pynative_synchronize=True)
    dataset_size = 10000
    num_epochs = 8
    dataset = ds.GeneratorDataset(DynamicData(dataset_size),
                                  ["input_ids", "input_mask", "segment_ids", "next_sentence_labels",
                                   "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"], num_parallel_workers=1)

    dataset_shapes, dataset_types = get_dataset_shapes_and_types(dataset)
    batch_size = dataset.get_batch_size()
    dataset = dataset.device_que()
    queue_name = dataset.queue_name
    _cell_graph_executor.init_dataset(queue_name, 1, batch_size, dataset_types, dataset_shapes, (), "")
    time.sleep(1)
    dataset.send(num_epochs)
    dataset_shapes = tuple([(-2,)] * len(dataset_shapes))
    get_next = P.GetNext(dataset_types, dataset_shapes, len(dataset_types), queue_name)
    start_memory = psutil.Process(os.getpid()).memory_info().rss
    for _ in range(num_epochs * dataset_size):
        get_next()
    end_memory = psutil.Process(os.getpid()).memory_info().rss
    assert end_memory < start_memory + 2147483648


if __name__ == '__main__':
    test_generatordataset_static_shape_memory()
    test_generatordataset_dynamic_shape_memory()

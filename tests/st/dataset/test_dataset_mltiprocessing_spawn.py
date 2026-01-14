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
# ==============================================================================
import os

import numpy as np
import pytest

import mindspore
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import nn, ops, Tensor
from mindspore.dataset.transforms import transforms
from tests.mark_utils import arg_mark


input_apple_jpg = "/home/workspace/mindspore_dataset/910B_dvpp/apple.jpg"


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize("python_multiprocessing", [False, True])
def test_generator_dataset_with_dvpp_with_spawn_independent_mode(python_multiprocessing):
    """
    Feature: GeneratorDataset op
    Description: Test GeneratorDataset with dvpp with multi process and map with dvpp with multi-thread + spawn mode
    by independent mode
    Expectation: The result is equal to the expected
    """

    os.environ['MS_INDEPENDENT_DATASET'] = "True"
    ds.config.set_multiprocessing_start_method("spawn")

    # Random-accessible object as input source
    class RandomAccessDataset:
        def __init__(self):
            self._data = np.ones((5, 2))
            self._label = np.zeros((5, 1))

        def __getitem__(self, index):
            image = np.fromfile(input_apple_jpg, dtype=np.uint8)
            image = vision.Decode()(image)
            image = vision.Crop((0, 0), 224)(image)
            image = vision.Resize((192, 192)).device("Ascend")(image)
            return image, self._label[index]

        def __len__(self):
            return len(self._data)

    def func(data):
        data = vision.Resize((128, 128)).device("Ascend")(data)
        return (data,)

    # map with multi process by spawn
    loader1 = RandomAccessDataset()
    dataset1 = ds.GeneratorDataset(source=loader1, column_names=["data", "label"], python_multiprocessing=True,
                                   num_parallel_workers=2)
    dataset1 = dataset1.map(func, input_columns=["data"], python_multiprocessing=python_multiprocessing,
                            num_parallel_workers=2)

    count = 0
    for _ in dataset1:
        count += 1
    assert count == 5

    os.environ['MS_INDEPENDENT_DATASET'] = "False"
    ds.config.set_multiprocessing_start_method("fork")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize("ms_independent_dataset", ["True", "False"])
def test_generator_dataset_and_map_with_dvpp_nn_ops_with_spawn(ms_independent_dataset):
    """
    Feature: GeneratorDataset and map op
    Description: Test GeneratorDataset and map with dvpp and ops and nn operator with multi process in spawn mode
    Expectation: The result is equal to the expected
    """
    os.environ['MS_INDEPENDENT_DATASET'] = ms_independent_dataset
    ds.config.set_multiprocessing_start_method("spawn")

    # Random-accessible object as input source
    class RandomAccessDataset:
        def __init__(self):
            self._data = np.ones((100, 2))
            self._label = np.zeros((100, 1))

        def __getitem__(self, index):
            image = np.fromfile(input_apple_jpg, dtype=np.uint8)
            image = vision.Decode()(image)
            image = vision.Crop((0, 0), 224)(image)
            image = vision.Resize((200, 200)).device("Ascend")(image)
            image = ops.dropout(Tensor(image, mindspore.float32), p=0.5)
            image = nn.PixelShuffle(2)(image)
            return image.asnumpy(), self._label[index]

        def __len__(self):
            return len(self._data)

    def func(data):
        # data.shape: (50, 400, 6)
        data = data.reshape((data.shape[1], -1, 3))
        data = vision.Resize((64, 64)).device("Ascend")(data)
        data = ops.dropout(Tensor(data), p=0.5)
        data = nn.PixelShuffle(2)(data)
        return (data.asnumpy(),)

    # map with multi process by spawn
    loader1 = RandomAccessDataset()
    dataset1 = ds.GeneratorDataset(source=loader1, column_names=["data", "label"], python_multiprocessing=True,
                                   num_parallel_workers=2)
    dataset1 = dataset1.map(func, input_columns=["data"], python_multiprocessing=True,
                            num_parallel_workers=2)

    count = 0
    for _ in dataset1:
        count += 1
    assert count == 100

    os.environ['MS_INDEPENDENT_DATASET'] = "False"
    ds.config.set_multiprocessing_start_method("fork")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
@pytest.mark.parametrize("python_multiprocessing", [True, False])
def test_generator_dataset_with_dvpp_with_spawn(python_multiprocessing):
    """
    Feature: GeneratorDataset op
    Description: Test GeneratorDataset with dvpp with multi process and map with dvpp with multi-thread in spawn mode
    Expectation: The result is equal to the expected
    """
    ds.config.set_multiprocessing_start_method("spawn")

    # Random-accessible object as input source
    class RandomAccessDataset:
        def __init__(self):
            self._data = np.ones((100, 2))
            self._label = np.zeros((100, 1))

        def __getitem__(self, index):
            image = np.fromfile(input_apple_jpg, dtype=np.uint8)
            image = vision.Decode()(image)
            image = vision.Crop((0, 0), 224)(image)
            image = vision.Resize((192, 192)).device("Ascend")(image)
            return image, self._label[index]

        def __len__(self):
            return len(self._data)

    def func(data):
        data = vision.Resize((128, 128)).device("Ascend")(data)
        return (data,)

    # map with multi process by spawn
    loader1 = RandomAccessDataset()
    dataset1 = ds.GeneratorDataset(source=loader1, column_names=["data", "label"], python_multiprocessing=True,
                                   num_parallel_workers=2)
    dataset1 = dataset1.map(func, input_columns=["data"], python_multiprocessing=python_multiprocessing,
                            num_parallel_workers=2)

    count = 0
    for _ in dataset1:
        count += 1
    assert count == 100

    # compose map with multi process by spawn
    loader2 = RandomAccessDataset()
    dataset2 = ds.GeneratorDataset(source=loader2, column_names=["data", "label"], python_multiprocessing=True,
                                   num_parallel_workers=2)
    transform_ops = [
        vision.Resize((128, 128)).device("Ascend")
        ]
    dataset2 = dataset2.map(transforms.Compose(transform_ops), input_columns=["data"],
                            python_multiprocessing=python_multiprocessing, num_parallel_workers=2)

    count = 0
    for _ in dataset2:
        count += 1
    assert count == 100
    ds.config.set_multiprocessing_start_method("fork")


if __name__ == '__main__':
    test_generator_dataset_with_dvpp_with_spawn_independent_mode(python_multiprocessing)
    test_generator_dataset_and_map_with_dvpp_nn_ops_with_spawn(ms_independent_dataset)
    test_generator_dataset_with_dvpp_with_spawn(python_multiprocessing)

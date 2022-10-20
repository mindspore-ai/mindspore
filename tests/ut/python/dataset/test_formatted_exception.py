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
# ==============================================================================
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.vision as vision


def test_generator_single_worker_exception():
    """
    Feature: Formatted exception.
    Description: Test formatted exception in GeneratorDataset scenario with one worker.
    Expectation: Python stack and summary message can be found in exception log.
    """
    class Gen():
        def __init__(self):
            self.data = [1, 2, 3, 4]
        def __getitem__(self, index):
            data = self.data[index]
            return data/0
        def __len__(self):
            return 4

    dataset = ds.GeneratorDataset(Gen(), ["image"], shuffle=False, num_parallel_workers=1)

    try:
        for data in dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
            print(data["image"].shape)
        assert False
    except RuntimeError as e:
        assert "Exception thrown from user defined Python function in dataset" in str(e)
        assert "Python Call Stack" in str(e)
        assert "Traceback (most recent call last):" in str(e)
        assert "ZeroDivisionError: division by zero" in str(e)
        assert "Dataset Pipeline Error Message:" in str(e)


def test_generator_multi_workers_exception():
    """
    Feature: Formatted exception.
    Description: Test formatted exception in GeneratorDataset scenario with multi-workers.
    Expectation: Python stack and summary message can be found in exception log.
    """
    def pyfunc(image):
        return image

    class Gen():
        def __init__(self):
            self.data = [[1], [2], [3], [4]]
        def __getitem__(self, index):
            image = Image.open(index)
            return image
        def __len__(self):
            return 4

    dataset = ds.GeneratorDataset(Gen(), ["image"], shuffle=False, num_parallel_workers=2)
    dataset = dataset.map(operations=pyfunc, input_columns=["image"])

    try:
        for data in dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
            print(data["image"].shape)
        assert False
    except RuntimeError as e:
        assert "Exception thrown from user defined Python function in dataset" in str(e)
        assert "Python Call Stack" in str(e)
        assert "Traceback (most recent call last):" in str(e)
        assert "NameError: name 'Image' is not defined" in str(e)
        assert "Dataset Pipeline Error Message:" in str(e)


def test_batch_operator_exception():
    """
    Feature: Formatted exception.
    Description: Test formatted exception in batch operator scenario.
    Expectation: Python stack and summary message can be found in exception log.
    """
    class Gen():
        def __init__(self):
            self.data = [np.ones((2)), np.ones((2)), np.ones((2)), np.ones((2, 3))]
        def __getitem__(self, index):
            return self.data[index]
        def __len__(self):
            return 4

    dataset = ds.GeneratorDataset(Gen(), ["image"], shuffle=False)
    dataset = dataset.batch(2)

    try:
        for data in dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
            print(data["image"].shape)
        assert False
    except RuntimeError as e:
        assert "Exception thrown from dataset pipeline. Refer to 'Dataset Pipeline Error Message'" in str(e)
        assert "Python Call Stack" not in str(e)
        assert "C++ Call Stack: (For framework developers)" in str(e)


def test_batch_operator_with_pyfunc_exception():
    """
    Feature: Formatted exception.
    Description: Test formatted exception in batch operator with pyfunc scenario.
    Expectation: Python stack and summary message can be found in exception log.
    """
    class Gen():
        def __init__(self):
            self.data = [np.ones((2)), np.ones((2)), np.ones((2)), np.ones((2))]
        def __getitem__(self, index):
            return self.data[index]
        def __len__(self):
            return 4

    def batch_func(col, batch_info):
        zero = 0
        fake_data = 1/zero
        return np.ones((3)), np.array(fake_data)

    dataset = ds.GeneratorDataset(Gen(), ["image"], shuffle=False)
    dataset = dataset.batch(2, per_batch_map=batch_func, input_columns=["image"])

    try:
        for data in dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
            print(data["image"].shape)
        assert False
    except RuntimeError as e:
        assert "Exception thrown from user defined Python function in dataset" in str(e)
        assert "Python Call Stack" in str(e)
        assert "Traceback (most recent call last):" in str(e)
        assert "in batch_func" in str(e)
        assert "Dataset Pipeline Error Message:" in str(e)


def test_map_operator_with_c_ops_and_multiprocessing_exception():
    """
    Feature: Formatted exception.
    Description: Test formatted exception in map operator with c ops scenario.
    Expectation: Python stack and summary message can be found in exception log.
    """
    class Gen():
        def __init__(self):
            self.data = [np.ones((10, 10, 3)),
                         np.ones((15, 15, 3)),
                         np.ones((5, 5, 3))]
        def __getitem__(self, index):
            return self.data[index]
        def __len__(self):
            return 3

    dataset = ds.GeneratorDataset(Gen(), ["image"], shuffle=False, num_parallel_workers=2)
    dataset = dataset.map(operations=vision.RandomCrop((8, 8)), input_columns=["image"], num_parallel_workers=2)

    try:
        for data in dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
            print(data["image"].shape)
        assert False
    except RuntimeError as e:
        assert "Shape is incorrect" in str(e)
        assert "Python Call Stack" not in str(e)
        assert "Dataset Pipeline Error Message:" in str(e)


def test_map_operator_with_pyfunc_and_multithreading_exception():
    """
    Feature: Formatted exception.
    Description: Test formatted exception in map operator with pyfunc scenario.
    Expectation: Python stack and summary message can be found in exception log.
    """
    def pyfunc(image):
        a = 1
        b = 0
        c = a/b
        return c

    class Gen():
        def __init__(self):
            self.data = [[1], [2], [3], [4]]
        def __getitem__(self, index):
            return self.data[index]
        def __len__(self):
            return 4

    dataset = ds.GeneratorDataset(Gen(), ["image"], shuffle=False, num_parallel_workers=2)
    dataset = dataset.map(operations=pyfunc, input_columns=["image"], num_parallel_workers=2)

    try:
        for data in dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
            print(data["image"].shape)
        assert False
    except RuntimeError as e:
        assert "Exception thrown from user defined Python function in dataset" in str(e)
        assert "Python Call Stack" in str(e)
        assert "Traceback (most recent call last):" in str(e)
        assert "Dataset Pipeline Error Message:" in str(e)


def test_map_operator_with_pyfunc_and_multiprocessing_exception():
    """
    Feature: Formatted exception.
    Description: Test formatted exception in map operator with pyfunc scenario.
    Expectation: Python stack and summary message can be found in exception log.
    """
    def pyfunc(image):
        a = 1
        b = 0
        c = a/b
        return c

    class Gen():
        def __init__(self):
            self.data = [[1], [2], [3], [4]]
        def __getitem__(self, index):
            return self.data[index]
        def __len__(self):
            return 4

    dataset = ds.GeneratorDataset(Gen(), ["image"], shuffle=False, num_parallel_workers=1)
    dataset = dataset.map(operations=pyfunc, input_columns=["image"], num_parallel_workers=2,
                          python_multiprocessing=True)

    try:
        for data in dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
            print(data["image"].shape)
        assert False
    except RuntimeError as e:
        assert "Exception thrown from user defined Python function in dataset" in str(e)
        assert "Python Call Stack" in str(e)
        assert "Traceback (most recent call last):" in str(e)
        assert "in pyfunc" in str(e)
        assert "Dataset Pipeline Error Message:" in str(e)

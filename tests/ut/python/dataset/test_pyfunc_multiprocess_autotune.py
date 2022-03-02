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
"""
Test Python Multiprocessing with AutoTuning
"""
import numpy as np
import pytest
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as c_transforms
import mindspore.dataset.transforms.py_transforms as py_transforms
import mindspore.dataset.vision.c_transforms as c_vision
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore.dataset.vision import Inter

DATA_DIR = "../data/dataset/testCifar10Data"


def create_pyfunc_dataset(batch_size=32, repeat_size=1, num_parallel_workers=1, num_samples=None):
    """
    Create Cifar10 dataset pipline with Map ops containing only Python functions and Python Multiprocessing enabled
    """

    # Define dataset
    cifar10_ds = ds.Cifar10Dataset(DATA_DIR, num_samples=num_samples)

    cifar10_ds = cifar10_ds.map(operations=[py_vision.ToType(np.int32)], input_columns="label",
                                num_parallel_workers=num_parallel_workers, python_multiprocessing=True)

    # Setup transforms list which include Python ops / Pyfuncs
    transforms_list = [
        py_vision.ToPIL(),
        py_vision.RandomGrayscale(prob=0.2),
        np.array]  # need to convert PIL image to a NumPy array to pass it to C++ operation
    compose_op = py_transforms.Compose(transforms_list)
    cifar10_ds = cifar10_ds.map(operations=compose_op, input_columns="image",
                                num_parallel_workers=num_parallel_workers,
                                python_multiprocessing=True)

    # Apply Dataset Ops
    buffer_size = 10000
    cifar10_ds = cifar10_ds.shuffle(buffer_size=buffer_size)
    cifar10_ds = cifar10_ds.batch(batch_size, drop_remainder=True)
    cifar10_ds = cifar10_ds.repeat(repeat_size)

    return cifar10_ds


def create_pyop_cop_dataset(batch_size=32, repeat_size=1, num_parallel_workers=1, num_samples=None):
    """
    Create Cifar10 dataset pipeline with Map ops containing just C Ops or just Pyfuncs
    """

    # Define dataset
    cifar10_ds = ds.Cifar10Dataset(DATA_DIR, num_samples=num_samples)

    # Map#1 - with Pyfunc
    cifar10_ds = cifar10_ds.map(operations=[py_vision.ToType(np.int32)], input_columns="label",
                                num_parallel_workers=num_parallel_workers, python_multiprocessing=True)

    # Map#2 - with C Ops
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081
    resize_op = c_vision.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    rescale_op = c_vision.Rescale(rescale, shift)
    rescale_nml_op = c_vision.Rescale(rescale_nml, shift_nml)
    hwc2chw_op = c_vision.HWC2CHW()
    transforms = [resize_op, rescale_op, rescale_nml_op, hwc2chw_op]
    compose_op = c_transforms.Compose(transforms)
    cifar10_ds = cifar10_ds.map(operations=compose_op, input_columns="image",
                                num_parallel_workers=num_parallel_workers,
                                python_multiprocessing=True)

    # Map#3 - with Pyfunc
    transforms_list = [lambda x: x]
    compose_op = py_transforms.Compose(transforms_list)
    cifar10_ds = cifar10_ds.map(operations=compose_op, input_columns="image",
                                num_parallel_workers=num_parallel_workers,
                                python_multiprocessing=True)

    # Apply Dataset Ops
    buffer_size = 10000
    cifar10_ds = cifar10_ds.shuffle(buffer_size=buffer_size)
    cifar10_ds = cifar10_ds.batch(batch_size, drop_remainder=True)
    cifar10_ds = cifar10_ds.repeat(repeat_size)

    return cifar10_ds


def create_mixed_map_dataset(batch_size=32, repeat_size=1, num_parallel_workers=1, num_samples=None):
    """
    Create Cifar10 dataset pipeline with a Map op containing of both C Ops and Pyfuncs
    """

    # Define dataset
    cifar10_ds = ds.Cifar10Dataset(DATA_DIR, num_samples=num_samples)

    cifar10_ds = cifar10_ds.map(operations=[py_vision.ToType(np.int32)], input_columns="label",
                                num_parallel_workers=num_parallel_workers, python_multiprocessing=True)

    # Map with operations: Pyfunc + C Ops + Pyfunc
    resize_op = c_vision.Resize((32, 32), interpolation=Inter.LINEAR)
    rescale_op = c_vision.Rescale(1.0 / 255.0, 0.0)
    rescale_nml_op = c_vision.Rescale(1 / 0.3081, -1 * 0.1307 / 0.3081)
    hwc2chw_op = c_vision.HWC2CHW()
    cifar10_ds = cifar10_ds.map(
        operations=[lambda x: x, resize_op, rescale_op, rescale_nml_op, hwc2chw_op, lambda y: y],
        input_columns="image", num_parallel_workers=num_parallel_workers,
        python_multiprocessing=True)

    # Apply Dataset Ops
    buffer_size = 10000
    cifar10_ds = cifar10_ds.shuffle(buffer_size=buffer_size)
    cifar10_ds = cifar10_ds.batch(batch_size, drop_remainder=True)
    cifar10_ds = cifar10_ds.repeat(repeat_size)

    return cifar10_ds


@pytest.mark.forked
class TestPythonMultiprocAutotune:

    def setup_method(self):
        """
        Run before each test function.
        """
        # Enable Dataset AutoTune
        self.original_autotune = ds.config.get_enable_autotune()
        ds.config.set_enable_autotune(True)

        # Reduce memory required by disabling the shared memory optimization
        self.mem_original = ds.config.get_enable_shared_mem()
        ds.config.set_enable_shared_mem(False)

    def teardown_method(self):
        """
        Run after each test function.
        """
        # Restore settings
        ds.config.set_enable_shared_mem(self.mem_original)
        ds.config.set_enable_autotune(self.original_autotune)

    @staticmethod
    def test_cifar10_pyfunc_pipeline():
        """
        Feature: Python Multiprocessing with AutoTune
        Description: Test pipeline with Map ops containing only Python function
        Expectation: Data pipeline executes successfully with correct number of rows
        """
        # Note: Set num_parallel_workers to minimum of 1
        mydata1 = create_pyfunc_dataset(32, 1, num_parallel_workers=1, num_samples=400)
        mycount1 = 0
        for _ in mydata1.create_dict_iterator(num_epochs=1):
            mycount1 += 1
        assert mycount1 == 12

    @staticmethod
    def test_cifar10_pyfunc_pipeline_all_samples():
        """
        Feature: Python Multiprocessing with AutoTune
        Description: Test pipeline with Map ops containing only Python function, with all samples in dataset
        Expectation: Data pipeline executes successfully with correct number of rows
        """
        # Note: Use all samples
        mydata1 = create_pyfunc_dataset(32, 1, num_parallel_workers=8)
        mycount1 = 0
        for _ in mydata1.create_dict_iterator(num_epochs=1):
            mycount1 += 1
        assert mycount1 == 312

    @staticmethod
    def test_cifar10_pyop_cop_pipeline():
        """
        Feature: Python Multiprocessing with AutoTune
        Description: Test pipeline with Map ops containing just C Ops or just Pyfuncs
        Expectation: Data pipeline executes successfully with correct number of rows
        """
        mydata1 = create_pyop_cop_dataset(16, 1, num_parallel_workers=4, num_samples=600)
        mycount1 = 0
        for _ in mydata1.create_dict_iterator(num_epochs=1):
            mycount1 += 1
        assert mycount1 == 37

    @staticmethod
    def test_cifar10_mixed_map_pipeline():
        """
        Feature: Python Multiprocessing with AutoTune
        Description: Test pipeline with a Map op containing of both C Ops and Pyfuncs
        Expectation: Data pipeline executes successfully with correct number of rows
        """
        mydata1 = create_mixed_map_dataset(32, 2, num_parallel_workers=12, num_samples=500)
        mycount1 = 0
        for _ in mydata1.create_dict_iterator(num_epochs=1):
            mycount1 += 1
        assert mycount1 == 30

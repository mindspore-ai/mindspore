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
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
from mindspore.dataset.vision import Inter

CIFAR10_DATA_DIR = "../data/dataset/testCifar10Data"
CIFAR100_DATA_DIR = "../data/dataset/testCifar100Data"


def create_pyfunc_dataset(batch_size=32, repeat_size=1, num_parallel_workers=1, num_samples=None):
    """
    Create Cifar10 dataset pipline with Map ops containing only Python functions and Python Multiprocessing enabled
    for Map ops and Batch op
    """

    # Define dataset
    cifar10_ds = ds.Cifar10Dataset(CIFAR10_DATA_DIR, num_samples=num_samples)

    cifar10_ds = cifar10_ds.map(operations=[vision.ToType(np.int32)], input_columns="label",
                                num_parallel_workers=num_parallel_workers, python_multiprocessing=True)

    # Setup transforms list which include Python implementations / Pyfuncs
    transforms_list = [
        vision.ToPIL(),
        vision.RandomGrayscale(prob=0.2),
        vision.ToNumpy()]  # need to convert PIL image to a NumPy array to pass it to C++ operation
    compose_op = transforms.Compose(transforms_list)
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
    Create Cifar10 dataset pipeline with Map ops containing just C Ops or just Pyfuncs, and
    Python Multiprocessing enabled for Map ops and Batch op
    """

    # Define dataset
    cifar10_ds = ds.Cifar10Dataset(CIFAR10_DATA_DIR, num_samples=num_samples)

    # Map#1 - with Pyfunc
    cifar10_ds = cifar10_ds.map(operations=[vision.ToType(np.int32)], input_columns="label",
                                num_parallel_workers=num_parallel_workers, python_multiprocessing=True)

    # Map#2 - with C Ops
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081
    resize_op = vision.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    rescale_op = vision.Rescale(rescale, shift)
    rescale_nml_op = vision.Rescale(rescale_nml, shift_nml)
    hwc2chw_op = vision.HWC2CHW()
    transform = [resize_op, rescale_op, rescale_nml_op, hwc2chw_op]
    compose_op = transforms.Compose(transform)
    cifar10_ds = cifar10_ds.map(operations=compose_op, input_columns="image",
                                num_parallel_workers=num_parallel_workers,
                                python_multiprocessing=True)

    # Map#3 - with Pyfunc
    transforms_list = [lambda x: x]
    compose_op = transforms.Compose(transforms_list)
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
    Create Cifar10 dataset pipeline with a Map op containing of both C Ops and Pyfuncs, and
    Python Multiprocessing enabled for Map ops and Batch op
    """

    # Define dataset
    cifar10_ds = ds.Cifar10Dataset(CIFAR10_DATA_DIR, num_samples=num_samples)

    cifar10_ds = cifar10_ds.map(operations=[vision.ToType(np.int32)], input_columns="label",
                                num_parallel_workers=num_parallel_workers, python_multiprocessing=True)

    # Map with operations: Pyfunc + C Ops + Pyfunc
    resize_op = vision.Resize((32, 32), interpolation=Inter.LINEAR)
    rescale_op = vision.Rescale(1.0 / 255.0, 0.0)
    rescale_nml_op = vision.Rescale(1 / 0.3081, -1 * 0.1307 / 0.3081)
    hwc2chw_op = vision.HWC2CHW()
    cifar10_ds = cifar10_ds.map(
        operations=[lambda x: x, resize_op, rescale_op, rescale_nml_op, hwc2chw_op, lambda y: y],
        input_columns="image", num_parallel_workers=num_parallel_workers, python_multiprocessing=True)

    # Apply Dataset Ops
    cifar10_ds = cifar10_ds.batch(batch_size, drop_remainder=True)
    cifar10_ds = cifar10_ds.repeat(repeat_size)

    return cifar10_ds


def create_per_batch_map_dataset(batch_size=32, repeat_size=1, num_parallel_workers=1, num_samples=None):
    """
    Create Cifar100 dataset pipline with Batch op using per_batch_map and Python Multiprocessing enabled
    """

    # Define dataset
    cifar100_ds = ds.Cifar100Dataset(CIFAR100_DATA_DIR, num_samples=num_samples)

    cifar100_ds = cifar100_ds.map(operations=[vision.ToType(np.int32)], input_columns="fine_label")

    cifar100_ds = cifar100_ds.map(operations=[lambda z: z], input_columns="image")

    # Callable function to delete 3rd column
    def del_column(col1, col2, col3, batch_info):
        return (col1, col2,)

    # Apply Dataset Ops
    buffer_size = 10000
    cifar100_ds = cifar100_ds.shuffle(buffer_size=buffer_size)
    # Note: Test repeat before batch
    cifar100_ds = cifar100_ds.repeat(repeat_size)
    cifar100_ds = cifar100_ds.batch(batch_size, per_batch_map=del_column,
                                    input_columns=['image', 'fine_label', 'coarse_label'],
                                    output_columns=['image', 'label'], drop_remainder=True,
                                    num_parallel_workers=num_parallel_workers, python_multiprocessing=True)

    return cifar100_ds


def create_mp_dataset(batch_size=32, repeat_size=1, num_parallel_workers=1, num_samples=None):
    """
    Create Cifar10 dataset pipline with Python Multiprocessing enabled for
    - Batch op using batch_per_map
    - Map ops using Pyfuncs
    """

    # Define dataset
    cifar10_ds = ds.Cifar10Dataset(CIFAR10_DATA_DIR, num_samples=num_samples)

    cifar10_ds = cifar10_ds.map(operations=[vision.ToType(np.int32)], input_columns="label",
                                num_parallel_workers=num_parallel_workers, python_multiprocessing=True)

    # Setup transforms list which include Python implementations / Pyfuncs
    transforms_list = [
        vision.ToPIL(),
        vision.RandomGrayscale(prob=0.8),
        vision.ToNumpy()]  # need to convert PIL image to a NumPy array to pass it to C++ operation
    compose_op = transforms.Compose(transforms_list)
    cifar10_ds = cifar10_ds.map(operations=compose_op, input_columns="image",
                                num_parallel_workers=num_parallel_workers, python_multiprocessing=True)

    # Callable function to swap columns
    def swap_columns(col1, col2, batch_info):
        return (col2, col1,)

    # Apply Dataset Ops
    buffer_size = 10000
    cifar10_ds = cifar10_ds.shuffle(buffer_size=buffer_size)
    cifar10_ds = cifar10_ds.batch(batch_size, drop_remainder=True,
                                  per_batch_map=swap_columns,
                                  input_columns=['image', 'label'],
                                  output_columns=['mylabel', 'myimage'],
                                  num_parallel_workers=num_parallel_workers,
                                  python_multiprocessing=True)
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
    def test_pymultiproc_at_map_pyfunc():
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
    def test_pymultiproc_at_map_pyfunc_pipeline_all_samples():
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
    def test_pymultiproc_at_map_pyop_cop():
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
    def test_pymultiproc_at_map_mixed():
        """
        Feature: Python Multiprocessing with AutoTune
        Description: Test pipeline with a Map op containing of both C Ops and Pyfuncs
        Expectation: Data pipeline executes successfully with correct number of rows
        """
        mydata1 = create_mixed_map_dataset(32, 2, num_parallel_workers=2, num_samples=500)
        mycount1 = 0
        for _ in mydata1.create_dict_iterator(num_epochs=1):
            mycount1 += 1
        assert mycount1 == 30

    @staticmethod
    def test_pymultiproc_at_per_batch_map():
        """
        Feature: Python Multiprocessing with AutoTune
        Description: Test pipeline with Batch op using per_batch_map
        Expectation: Data pipeline executes successfully with correct number of rows
        """
        # Note: Set num_parallel_workers to minimum of 1
        mydata1 = create_per_batch_map_dataset(32, repeat_size=3, num_parallel_workers=1, num_samples=300)
        mycount1 = 0
        for _ in mydata1.create_dict_iterator(num_epochs=1):
            mycount1 += 1
        assert mycount1 == 28

    @staticmethod
    def test_pymultiproc_at_pipeline():
        """
        Feature: Python Multiprocessing with AutoTune
        Description: Test pipeline with Python multiprocessing enabled for dataset, map and batch ops
        Expectation: Data pipeline executes successfully with correct number of rows
        """
        mydata1 = create_mp_dataset(32, repeat_size=2, num_parallel_workers=2, num_samples=700)
        mycount1 = 0
        for _ in mydata1.create_dict_iterator(num_epochs=1):
            mycount1 += 1
        assert mycount1 == 42


if __name__ == '__main__':
    TestPythonMultiprocAutotune.test_pymultiproc_at_map_pyfunc()
    TestPythonMultiprocAutotune.test_pymultiproc_at_map_pyfunc_pipeline_all_samples()
    TestPythonMultiprocAutotune.test_pymultiproc_at_map_pyop_cop()
    TestPythonMultiprocAutotune.test_pymultiproc_at_map_mixed()
    TestPythonMultiprocAutotune.test_pymultiproc_at_per_batch_map()
    TestPythonMultiprocAutotune.test_pymultiproc_at_pipeline()

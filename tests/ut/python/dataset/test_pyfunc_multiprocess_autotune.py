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

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore.dataset.vision import Inter

CIFAR10_DATA_DIR = "../data/dataset/testCifar10Data"
CIFAR100_DATA_DIR = "../data/dataset/testCifar100Data"






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
    Create Cifar100 dataset pipeline with Batch op using per_batch_map and Python Multiprocessing enabled
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

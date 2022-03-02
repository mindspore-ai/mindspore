# Copyright 2021 Huawei Technologies Co., Ltd
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
import time
import pytest
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.transforms.py_transforms as py_transforms
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.vision.py_transforms as py_vision

from mindspore import context, nn
from mindspore.common import dtype as mstype, set_seed
from mindspore.dataset.vision import Inter
from mindspore.train import Model


def create_model():
    """
    Define and return a simple model
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            return x

    net = Net()
    model_simple = Model(net)

    return model_simple


def create_dataset(data_path, batch_size=32, repeat_size=1, num_parallel_workers=1):
    """
    Create dataset for train or test
    """
    # Define dataset
    mnist_ds = ds.MnistDataset(data_path)

    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # Define map operations
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # Apply map operations on images
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_nml_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)

    # Apply DatasetOps
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.forked
def test_autotune_train_simple_model():
    """
    Feature: Dataset AutoTune
    Description: Test Dataset AutoTune for Training of a Simple Model
    Expectation: Training completes successfully

    """
    original_seed = ds.config.get_seed()
    set_seed(1)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    context.set_context(enable_graph_kernel=True)

    # Enable Dataset AutoTune
    original_autotune = ds.config.get_enable_autotune()
    ds.config.set_enable_autotune(True)

    ds_train = create_dataset(os.path.join("/home/workspace/mindspore_dataset/mnist", "train"), 32, 1)
    model = create_model()

    print("Start Training.")
    epoch_size = 10
    model.train(epoch_size, ds_train)
    print("Training is finished.")

    # Restore settings
    ds.config.set_enable_autotune(original_autotune)
    ds.config.set_seed(original_seed)


def create_dataset_pyfunc_multiproc(data_path, batch_size=32, num_map_parallel_workers=1, max_rowsize=16):
    """
    Create dataset with Python ops list and python_multiprocessing=True for Map op
    """

    # Define dataset
    data1 = ds.MnistDataset(data_path, num_parallel_workers=8)

    data1 = data1.map(operations=[py_vision.ToType(np.int32)], input_columns="label",
                      num_parallel_workers=num_map_parallel_workers,
                      python_multiprocessing=True, max_rowsize=max_rowsize)

    # Setup transforms list which include Python ops
    transforms_list = [
        py_vision.ToTensor(),
        lambda x: x,
        py_vision.HWC2CHW(),
        py_vision.RandomErasing(0.9, value='random'),
        py_vision.Cutout(4, 2),
        lambda y: y
    ]
    compose_op = py_transforms.Compose(transforms_list)
    data1 = data1.map(operations=compose_op, input_columns="image", num_parallel_workers=num_map_parallel_workers,
                      python_multiprocessing=True, max_rowsize=max_rowsize)

    # Apply Dataset Ops
    data1 = data1.batch(batch_size, drop_remainder=True)

    return data1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.forked
def test_autotune_pymultiproc_train_simple_model():
    """
    Feature: Dataset AutoTune
    Description: Test Dataset AutoTune with Python Multiprocessing for Training of a Simple Model
    Expectation: Training completes successfully

    """
    original_seed = ds.config.get_seed()
    set_seed(20)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    context.set_context(enable_graph_kernel=True)

    # Reduce memory required by disabling the shared memory optimization
    mem_original = ds.config.get_enable_shared_mem()
    ds.config.set_enable_shared_mem(False)

    # Enable Dataset AutoTune
    original_autotune = ds.config.get_enable_autotune()
    ds.config.set_enable_autotune(True)
    original_interval = ds.config.get_autotune_interval()
    ds.config.set_autotune_interval(100)

    ds_train = create_dataset_pyfunc_multiproc(os.path.join("/home/workspace/mindspore_dataset/mnist", "train"), 32, 2)
    model = create_model()

    print("Start Model Training.")
    model_start = time.time()
    epoch_size = 2
    model.train(epoch_size, ds_train)
    print("Model training is finished. Took {}s".format(time.time() - model_start))

    # Restore settings
    ds.config.set_autotune_interval(original_interval)
    ds.config.set_enable_autotune(original_autotune)
    ds.config.set_enable_shared_mem(mem_original)
    ds.config.set_seed(original_seed)


if __name__ == "__main__":
    test_autotune_train_simple_model()
    test_autotune_pymultiproc_train_simple_model()

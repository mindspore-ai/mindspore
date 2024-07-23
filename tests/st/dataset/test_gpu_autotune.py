# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision

from mindspore import context, nn
from mindspore.common import dtype as mstype, set_seed
from mindspore.dataset.vision import Inter
from mindspore.train import Model
from tests.mark_utils import arg_mark


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


def create_dataset(data_path, batch_size=32, num_parallel_workers=1):
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
    resize_op = vision.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    rescale_nml_op = vision.Rescale(rescale_nml, shift_nml)
    rescale_op = vision.Rescale(rescale, shift)
    hwc2chw_op = vision.HWC2CHW()
    type_cast_op = transforms.TypeCast(mstype.int32)

    # Apply map operations on images
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_nml_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)

    return mnist_ds


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_autotune_train_simple_model(tmp_path):
    """
    Feature: Dataset AutoTune
    Description: Test Dataset AutoTune for training of a simple model and deserialize the written at config file
    Expectation: Training and data deserialization completes successfully

    """
    rank_id = os.getenv("RANK_ID")
    if not rank_id or not rank_id.isdigit():
        rank_id = "0"

    original_seed = ds.config.get_seed()
    set_seed(1)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    context.set_context(enable_graph_kernel=True)
    at_config_filename = "test_autotune_train_simple_model_at_config"

    # Enable Dataset AutoTune
    original_autotune = ds.config.get_enable_autotune()
    ds.config.set_enable_autotune(True, str(tmp_path / at_config_filename))

    ds_train = create_dataset(os.path.join("/home/workspace/mindspore_dataset/mnist", "train"), 32)
    model = create_model()

    print("Start training.")
    epoch_size = 10
    start_time = time.time()
    model.train(epoch_size, ds_train, dataset_sink_mode=True)
    print("Training finished. Took {}s".format(time.time() - start_time))

    ds.config.set_enable_autotune(False)

    file = tmp_path / (at_config_filename + "_" + rank_id + ".json")
    assert file.exists()
    ds_train_deserialized = ds.deserialize(json_filepath=str(file))

    num = 0
    for data1, data2 in zip(ds_train.create_dict_iterator(num_epochs=1, output_numpy=True),
                            ds_train_deserialized.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(data1['image'], data2['image'])
        np.testing.assert_array_equal(data1['label'], data2['label'])
        num += 1

    assert num == 1875

    # Restore settings
    ds.config.set_enable_autotune(original_autotune)
    ds.config.set_seed(original_seed)


def create_dataset_pyfunc_multiproc(data_path, batch_size=32, num_op_parallel_workers=1, max_rowsize=16):
    """
    Create Mnist dataset pipline with Python Multiprocessing enabled for
    - Batch op using per_batch_map
    - Map ops using Pyfuncs
    """

    # Define dataset with num_parallel_workers=8 for reasonable performance
    data1 = ds.MnistDataset(data_path, num_parallel_workers=8)

    data1 = data1.map(operations=[vision.ToType(np.int32)], input_columns="label",
                      num_parallel_workers=num_op_parallel_workers,
                      python_multiprocessing=True, max_rowsize=max_rowsize)

    # Setup transforms list which include Python ops
    transforms_list = [
        lambda x: x,
        vision.HWC2CHW(),
        vision.RandomErasing(0.9, value='random'),
        lambda y: y
    ]
    compose_op = transforms.Compose(transforms_list)
    data1 = data1.map(operations=compose_op, input_columns="image", num_parallel_workers=num_op_parallel_workers,
                      python_multiprocessing=True, max_rowsize=max_rowsize)

    # Callable function to swap order of 2 columns
    def swap_columns(col1, col2, batch_info):
        return (col2, col1,)

    # Apply Dataset Ops
    data1 = data1.batch(batch_size, drop_remainder=True, per_batch_map=swap_columns,
                        input_columns=['image', 'label'],
                        output_columns=['mylabel', 'myimage'],
                        num_parallel_workers=num_op_parallel_workers, python_multiprocessing=True)

    return data1


@pytest.mark.skip(reason="get_next time out")
@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='essential')
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
    model.train(epoch_size, ds_train, dataset_sink_mode=True)
    print("Model training is finished. Took {}s".format(time.time() - model_start))

    # Restore settings
    ds.config.set_autotune_interval(original_interval)
    ds.config.set_enable_autotune(original_autotune)
    ds.config.set_enable_shared_mem(mem_original)
    ds.config.set_seed(original_seed)


if __name__ == "__main__":
    test_autotune_train_simple_model("")
    test_autotune_pymultiproc_train_simple_model()

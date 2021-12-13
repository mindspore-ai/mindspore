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
import pytest

import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV

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
def test_autotune_train_simple_model():
    """
    Feature: Dataset AutoTune
    Description: Test Dataset AutoTune for Training of a Simple Model
    Expectation: Training completes successfully

    """
    set_seed(1)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    context.set_context(enable_graph_kernel=True)

    # Enable Dataset AutoTune
    ds.config.set_enable_autotune(True)

    ds_train = create_dataset(os.path.join("/home/workspace/mindspore_dataset/mnist", "train"), 32, 1)
    model = create_model()

    print("Start Training.")
    epoch_size = 10
    model.train(epoch_size, ds_train)
    print("Training is finished.")

    # Disable Dataset AutoTune
    ds.config.set_enable_autotune(False)

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
# ============================================================================

import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV
import mindspore.nn as nn
from mindspore.common import dtype as mstype
from mindspore.dataset.vision import Inter
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import operations as P
from mindspore import Tensor


def create_dataset(data_path, batch_size=32, repeat_size=1, rank_id=0, rank_size=1,
                   num_parallel_workers=1):
    """
    create dataset for train or test
    """
    # define dataset
    mnist_ds = ds.MnistDataset(data_path, num_shards=rank_size, shard_id=rank_id)

    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # define map operations
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)  # Bilinear mode
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # apply map operations on images
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_nml_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)  # 10000 as in LeNet train script
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds


def fc_with_initialize(input_channels, out_channels):
    """weight initial for fc layer"""
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


def weight_variable():
    """weight initial"""
    return TruncatedNormal(0.02)


class SplitRefWithoutOptimNet(nn.Cell):
    def __init__(self, num_class=10, dist=False):
        super(SplitRefWithoutOptimNet, self).__init__()
        self.num_class = num_class
        self.fc1 = fc_with_initialize(1024, 120)
        self.weight1 = self.fc1.weight
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, self.num_class)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.assign_add = P.AssignAdd()

        self.assign_value = Tensor(np.ones([120, 1024]).astype(np.float32) * 0.1)
        # To cover the scenario that splitting side-effect nodes in hetegeneous case.
        self.assign_add.add_prim_attr("primitive_target", "CPU")
        if dist:
            self.assign_add.place("MS_WORKER", 1)

    def construct(self, x):
        x = self.flatten(x)
        self.assign_add(self.weight1, self.assign_value)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class SplitOptimNet(nn.Cell):
    def __init__(self, num_class=10, dist=False):
        super(SplitOptimNet, self).__init__()
        self.num_class = num_class
        self.fc1 = fc_with_initialize(1024, 120)
        self.weight1 = self.fc1.weight
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, self.num_class)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.assign_add = P.AssignAdd()

        self.assign_value = Tensor(np.ones([120, 1024]).astype(np.float32) * 0.1)
        # To cover the scenario that splitting side-effect nodes in hetegeneous case.
        self.assign_add.add_prim_attr("primitive_target", "CPU")

    def construct(self, x):
        x = self.flatten(x)
        self.assign_add(self.weight1, self.assign_value)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def get_optimizer(net, lr=0.01, dist=False):
    momentum = 0.9
    mom_optimizer = nn.Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, momentum)
    # To cover the scenario that splitting side-effect nodes in hetegeneous case.
    mom_optimizer.opt.add_prim_attr("primitive_target", "CPU")
    if dist:
        mom_optimizer.place("MS_WORKER", 1)
    return mom_optimizer


def get_loss():
    return nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')


def get_dataset(data_path, rank_id=0, rank_size=32):
    return create_dataset(data_path, 32, 1, rank_id, rank_size)

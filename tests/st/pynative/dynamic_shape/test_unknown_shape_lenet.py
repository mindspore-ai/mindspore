# Copyright 2019-2022 Huawei Technologies Co., Ltd
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

import numpy as np

import mindspore.context as context
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.dataset.vision import Inter
from mindspore.nn import Dense, TrainOneStepCell, WithLossCell
from mindspore.train import Accuracy
from mindspore.nn.optim import Momentum
from mindspore.ops import operations as P
from mindspore.train import Model, Callback
from mindspore.common.initializer import TruncatedNormal
from mindspore.train.callback._callback import _handle_loss
from mindspore.common import JitConfig
from mindspore.common import set_seed
from tests.mark_utils import arg_mark


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """weight initial for conv layer"""
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    """weight initial for fc layer"""
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


def weight_variable():
    """weight initial"""
    return TruncatedNormal(0.02)


class LeNet5(nn.Cell):
    def __init__(self, num_class=10, channel=1):
        super(LeNet5, self).__init__()
        self.num_class = num_class
        self.conv1 = conv(channel, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, self.num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class LeNet(nn.Cell):
    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = P.ReLU()
        self.batch_size = 1
        weight1 = Tensor(np.ones([6, 3, 5, 5]).astype(np.float32) * 0.01)
        weight2 = Tensor(np.ones([16, 6, 5, 5]).astype(np.float32) * 0.01)
        self.conv1 = nn.Conv2d(3, 6, (5, 5), weight_init=weight1, stride=1, padding=0, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, (5, 5), weight_init=weight2, pad_mode='valid', stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="valid")

        self.reshape = P.Reshape()
        self.reshape1 = P.Reshape()

        self.fc1 = Dense(400, 120)
        self.fc2 = Dense(120, 84)
        self.fc3 = Dense(84, 10)

    def construct(self, input_x):
        output = self.conv1(input_x)
        output = self.relu(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.pool(output)
        output = self.reshape(output, (self.batch_size, -1))
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output


def multisteplr(total_steps, gap, base_lr=0.9, gamma=0.1, dtype=mstype.float32):
    lr = []
    for step in range(total_steps):
        lr_ = base_lr * gamma ** (step // gap)
        lr.append(lr_)
    return Tensor(np.array(lr), dtype)


def create_dataset(data_path, batch_size=32, repeat_size=1, num_parallel_workers=1, num_samples=None,
                   enable_shuffle=True):
    """
    create dataset for train or test
    """
    # define dataset
    if num_samples is None:
        mnist_ds = ds.MnistDataset(data_path)
    else:
        mnist_ds = ds.MnistDataset(data_path, num_samples=num_samples)

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
    if enable_shuffle:
        mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)  # 10000 as in LeNet train script
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds


class CustomCallback(Callback):
    """
    Assert the loss in every step_end.
    """

    def __init__(self):
        super(CustomCallback, self).__init__()
        self.except_loss = [2.300181, 2.305321, 2.304834, 2.297870, 2.303635]

    def step_end(self, run_context):
        """
        Print training loss at the end of step.

        Args:
            run_context (RunContext): Include some information of the model.  For more details,
                    please refer to :class:`mindspore.train.RunContext`.
        """
        cb_params = run_context.original_args()
        loss = _handle_loss(cb_params.net_outputs)
        print("loss is ", loss, flush=True)
        assert np.allclose(self.except_loss[cb_params.cur_step_num - 1], loss)


@arg_mark(plat_marks=['platform_gpu'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_set_inputs_cell_before_top_cell_unknown_shape():
    """
    Feature: PyNative Unknown shape
    Description: C1.set_inputs, run C1(x); C2 is top cell, and run C2(x).
    Expectation: No exception.
    """
    set_seed(1)
    np.random.seed(1)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    ds_train = create_dataset(os.path.join("/home/workspace/mindspore_dataset/mnist", "train"), 32, 1, 1, 160,
                              False)
    network = LeNet5(10)
    network.set_jit_config(JitConfig(jit_level="O2"))

    input_x = Tensor(shape=[None, 1, 32, 32], dtype=mstype.float32)
    network.set_inputs(input_x)

    input_label = Tensor(shape=[None,], dtype=mstype.int32)
    input_logits = Tensor(shape=[None, 10], dtype=mstype.float32)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_loss.set_inputs(input_logits, input_label)

    net_opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    # First, run SoftmaxCrossEntropyWithLogits set_inputs
    # Second, top cell is WithLossCell
    model.train(1, ds_train, callbacks=[CustomCallback()], dataset_sink_mode=True)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_set_inputs_cell_after_top_cell_unknown_shape():
    """
    Feature: PyNative Unknown shape
    Description: C1 is top cell, run C1(x); C2 set_inputs, and run C2(x)
    Expectation: No exception.
    """
    net = LeNet()
    # set inputs unknown shape
    input_x = Tensor(shape=[None, 3, 32, 32], dtype=mstype.float32)
    net.set_inputs(input_x)

    momentum = 0.9
    learning_rate = multisteplr(10, 30)

    optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate, momentum)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)  # optimizer
    train_network.set_train()
    for _ in range(2):
        data = Tensor(np.ones([net.batch_size, 3, 32, 32]).astype(np.float32) * 0.01)
        label = Tensor(np.ones([net.batch_size]).astype(np.int32))

        # First, top cell is WithLossCell
        # Second, net run with set_inputs
        train_network(data, label).asnumpy()

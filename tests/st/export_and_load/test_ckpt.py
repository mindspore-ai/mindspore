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
import os
import stat

import numpy as np
import pytest

import mindspore.context as context
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV
import mindspore.nn as nn
from mindspore.train import ModelCheckpoint, CheckpointConfig, Callback, LossMonitor
from mindspore import load_checkpoint
from mindspore.common import dtype as mstype
from mindspore.dataset.vision import Inter
from mindspore.train import Model
from mindspore.common.initializer import TruncatedNormal
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.nn import Accuracy
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


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


def create_dataset(data_path, batch_size=32, repeat_size=1, num_parallel_workers=1):
    """create dataset for train or test"""
    mnist_ds = ds.MnistDataset(data_path)

    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)  # Bilinear mode
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

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


class ErrorCallback(Callback):
    def __init__(self, epoch_num):
        self.epoch_num = epoch_num

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        if epoch_num == self.epoch_num:
            raise RuntimeError("Exec runtime error.")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_ckpt_append_info():
    """
    Feature: Save append info during save ckpt.
    Description: Save append info during save ckpt.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    network = LeNet5(10)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)
    model = Model(network, net_loss, net_opt)
    ds_train = create_dataset(os.path.join('/home/workspace/mindspore_dataset/mnist', "train"), 32, 1)
    cb_config = CheckpointConfig(save_checkpoint_steps=ds_train.get_dataset_size(),
                                 append_info=["epoch_num", "step_num"])
    ckpoint_cb = ModelCheckpoint(prefix='append_info', directory="./", config=cb_config)
    model.train(3, ds_train, callbacks=ckpoint_cb, dataset_sink_mode=True)
    file_list = os.listdir(os.getcwd())
    ckpt_list = [k for k in file_list if k.startswith("append_info")]
    ckpt_1 = [k for k in ckpt_list if k.startswith("append_info-2")]
    dict_1 = load_checkpoint(ckpt_1[0])
    assert dict_1.get("epoch_num").data.asnumpy() == 2
    for file_name in ckpt_list:
        if os.path.exists(file_name):
            os.chmod(file_name, stat.S_IWRITE)
            os.remove(file_name)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_ckpt_error():
    """
    Feature: Save ckpt when error in train.
    Description: Save ckpt when error in train.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    network = LeNet5(10)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)
    model = Model(network, net_loss, net_opt)

    print("============== Starting Training ==============")
    ds_train = create_dataset(os.path.join('/home/workspace/mindspore_dataset/mnist', "train"), 32, 1)
    cb_config = CheckpointConfig(save_checkpoint_steps=10000, exception_save=True)
    ckpoint_cb = ModelCheckpoint(prefix='error_ckpt', directory="./", config=cb_config)
    with pytest.raises(RuntimeError):
        model.train(3, ds_train, callbacks=[ckpoint_cb, ErrorCallback(2)], dataset_sink_mode=True)
    file_list = os.listdir(os.getcwd())
    ckpt_list = [k for k in file_list if k.endswith("_breakpoint.ckpt")]
    assert os.path.exists(ckpt_list[0])
    if os.path.exists(ckpt_list[0]):
        os.chmod(ckpt_list[0], stat.S_IWRITE)
        os.remove(ckpt_list[0])


def get_data(num, img_size=(1, 32, 32), num_classes=10, is_onehot=True):
    """Get Data"""
    for _ in range(num):
        img = np.random.randn(*img_size)
        target = np.random.randint(0, num_classes)
        target_ret = np.array([target]).astype(np.float32)
        if is_onehot:
            target_onehot = np.zeros(shape=(num_classes,))
            target_onehot[target] = 1
            target_ret = target_onehot.astype(np.float32)
        yield img.astype(np.float32), target_ret


def create_dataset_lenet(num_data=32, batch_size=32, repeat_size=1):
    """Generate Data"""
    input_data = ds.GeneratorDataset(list(get_data(num_data)), column_names=['data', 'label'])
    input_data = input_data.batch(batch_size, drop_remainder=True)
    input_data = input_data.repeat(repeat_size)
    return input_data


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_checkpointconfig_append_info_and_load_checkpoint():
    """
    Feature: Save checkpoint for CheckpointConfig's append_info and load checkpoint.
    Description: Test save checkpoint and load checkpoint for CheckpointConfig's append_info.
    Expectation: Checkpoint for CheckpointConfig's append_info can be saved and reloaded.
    """
    if os.path.exists('./ckptconfig_append_info-1_1.ckpt'):
        os.chmod('./ckptconfig_append_info-1_1.ckpt', stat.S_IWRITE)
        os.remove('./ckptconfig_append_info-1_1.ckpt')

    ds_train = create_dataset_lenet()
    network = LeNet5(10)
    net_loss = nn.SoftmaxCrossEntropyWithLogits()
    net_opt = nn.Momentum(network.trainable_params(), learning_rate=0.01, momentum=0.9)
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
    config_ck = CheckpointConfig(save_checkpoint_steps=1, keep_checkpoint_max=1,
                                 append_info=[{'param_1': Tensor(200.0),
                                               'param_2': Parameter(Tensor([[1, 2], [3, 4]])),
                                               'param_3': 'param_string'}])
    ckpt_cb = ModelCheckpoint(prefix='ckptconfig_append_info', directory='./', config=config_ck)
    model.train(epoch=1, train_dataset=ds_train, callbacks=[LossMonitor(), ckpt_cb])

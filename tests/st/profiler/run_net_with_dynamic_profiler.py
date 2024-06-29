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
from argparse import ArgumentParser

import mindspore as ms
from mindspore import dataset as ds
from mindspore import nn, Tensor, context
from mindspore.train import Accuracy
from mindspore.nn.optim import Momentum
from mindspore.dataset.transforms import transforms as C
from mindspore.dataset.vision import transforms as CV
from mindspore.dataset.vision import Inter
from mindspore.common import dtype as mstype
from mindspore.common.initializer import TruncatedNormal
from mindspore.train import Model
import shutil


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
    """Define LeNet5 network."""

    def __init__(self, num_class=10, channel=1):
        """Net init."""
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
        self.channel = Tensor(channel)

    def construct(self, data):
        """define construct."""
        output = self.conv1(data)
        output = self.relu(output)
        output = self.max_pool2d(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.max_pool2d(output)
        output = self.flatten(output)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        return output


def create_dataset(data_path, batch_size=32, repeat_size=1, num_parallel_workers=1):
    """create dataset for train"""
    # define dataset
    mnist_ds = ds.MnistDataset(data_path, num_samples=batch_size * 5)

    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # define map operations
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)  # Bilinear mode
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift=0.0)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # apply map operations on images
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_nml_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds


def merge_folders(source_folder, target_folder):
    """
    Move current folder to target folder.
    """

    for item in os.listdir(source_folder):
        source = os.path.join(source_folder, item)
        target = os.path.join(target_folder, item)

        if os.path.exists(target):
            if os.path.exists(source):
                merge_folders(source, target)
            else:
                print(f"The file {item} is exist in {target_folder}")
        else:
            shutil.move(source, target)


class DynamicProfiler(ms.Callback):
    """
    Dynamic Profiler callback.
    Args:
        data_path (str): Data storage address.
    """

    def __init__(self, data_path):
        super(DynamicProfiler, self).__init__()
        self.rank_id = int(os.getenv('RANK_ID', '0'))
        self.output_path = data_path
        self.profiler = ms.Profiler(start_profile=False,
                                    output_path=os.path.join(self.output_path, 'current', str(self.rank_id)))

    def on_train_step_begin(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        self.profiler.start()

        step_path = os.path.join(self.output_path, str(step_num))
        if not os.path.exists(step_path):
            os.makedirs(step_path)

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        self.profiler.stop()
        merge_folders(os.path.join(self.output_path, 'current', str(self.rank_id)),
                      os.path.join(self.output_path, str(step_num), str(self.rank_id)))


def train_with_profiler():
    """Train Net with profiling."""
    target = args.target
    mode = args.mode
    output_path = args.output_path
    mnist_path = '/home/workspace/mindspore_dataset/mnist'
    context.set_context(mode=mode, device_target=target)
    ds_train = create_dataset(os.path.join(mnist_path, "train"))
    if ds_train.get_dataset_size() == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")

    lenet = LeNet5()
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    optim = Momentum(lenet.trainable_params(), learning_rate=0.1, momentum=0.9)
    profile_callback = DynamicProfiler(output_path)

    model = Model(lenet, loss_fn=loss, optimizer=optim, metrics={'acc': Accuracy()})
    model.train(3, ds_train, callbacks=[profile_callback], dataset_sink_mode=False)


parser = ArgumentParser(description='test dynamic profiler')
parser.add_argument('--target', type=str)
parser.add_argument('--mode', type=int)
parser.add_argument('--output_path', type=str)
args = parser.parse_args()
train_with_profiler()

# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
import shutil

import sys

from tests.security_utils import security_off_wrap
import pytest

from mindspore import dataset as ds
from mindspore import nn, Tensor, context
from mindspore.train.metrics import Accuracy
from mindspore.nn.optim import Momentum
from mindspore.dataset.transforms import transforms as C
from mindspore.dataset.vision import transforms as CV
from mindspore.dataset.vision import Inter
from mindspore.common import dtype as mstype
from mindspore.common.initializer import TruncatedNormal
from mindspore.train import Model
from mindspore import Profiler


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
    mnist_ds = ds.MnistDataset(data_path, num_samples=batch_size * 10)

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


def cleanup():
    data_path = os.path.join(os.getcwd(), "data")
    kernel_meta_path = os.path.join(os.getcwd(), "kernel_data")
    cache_path = os.path.join(os.getcwd(), "__pycache__")
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    if os.path.exists(kernel_meta_path):
        shutil.rmtree(kernel_meta_path)
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)


class TestProfiler:
    device_id = int(os.getenv('DEVICE_ID')) if os.getenv('DEVICE_ID') else 0
    rank_id = int(os.getenv('RANK_ID')) if os.getenv('RANK_ID') else 0
    mnist_path = '/home/workspace/mindspore_dataset/mnist'

    @classmethod
    def setup_class(cls):
        """Run begin all test case start."""
        cleanup()

    @staticmethod
    def teardown():
        """Run after each test case end."""
        cleanup()

    @pytest.mark.level2
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @security_off_wrap
    def test_cpu_profiler(self):
        if sys.platform != 'linux':
            return
        self._train_with_profiler(device_target="CPU", profile_memory=False)
        self._check_cpu_profiling_file()

    @pytest.mark.level1
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    @security_off_wrap
    def test_gpu_profiler(self):
        self._train_with_profiler(device_target="GPU", profile_memory=False)
        self._check_gpu_profiling_file()

    @pytest.mark.level1
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    @security_off_wrap
    def test_gpu_profiler_pynative(self):
        """
        Feature: profiler support GPU pynative mode.
        Description: profiling l2 GPU pynative mode data, analyze performance issues.
        Expectation: No exception.
        """
        self._train_with_profiler(device_target="GPU", profile_memory=False, context_mode=context.PYNATIVE_MODE)
        self._check_gpu_profiling_file()

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    @security_off_wrap
    def test_ascend_profiler(self):
        self._train_with_profiler(device_target="Ascend", profile_memory=True)
        self._check_d_profiling_file()

    def _train_with_profiler(self, device_target, profile_memory, context_mode=context.GRAPH_MODE):
        context.set_context(mode=context_mode, device_target=device_target)
        ds_train = create_dataset(os.path.join(self.mnist_path, "train"))
        if ds_train.get_dataset_size() == 0:
            raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")

        profiler = Profiler(profile_memory=profile_memory, output_path='data')
        profiler_name = os.listdir(os.path.join(os.getcwd(), 'data'))[0]
        self.profiler_path = os.path.join(os.getcwd(), f'data/{profiler_name}/')
        lenet = LeNet5()
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        optim = Momentum(lenet.trainable_params(), learning_rate=0.1, momentum=0.9)
        model = Model(lenet, loss_fn=loss, optimizer=optim, metrics={'acc': Accuracy()})

        model.train(1, ds_train, dataset_sink_mode=True)
        profiler.analyse()
        if device_target != 'Ascend':
            profiler.op_analyse(op_name="Conv2D")

    def _check_gpu_profiling_file(self):
        op_detail_file = self.profiler_path + f'gpu_op_detail_info_{self.device_id}.csv'
        op_type_file = self.profiler_path + f'gpu_op_type_info_{self.device_id}.csv'
        activity_file = self.profiler_path + f'gpu_activity_data_{self.device_id}.csv'
        timeline_file = self.profiler_path + f'gpu_timeline_display_{self.device_id}.json'
        getnext_file = self.profiler_path + f'minddata_getnext_profiling_{self.device_id}.txt'
        pipeline_file = self.profiler_path + f'minddata_pipeline_raw_{self.device_id}.csv'
        framework_file = self.profiler_path + f'gpu_framework_{self.device_id}.txt'

        gpu_profiler_files = (op_detail_file, op_type_file, activity_file,
                              timeline_file, getnext_file, pipeline_file, framework_file)
        for file in gpu_profiler_files:
            assert os.path.isfile(file)

    def _check_d_profiling_file(self):
        aicore_file = self.profiler_path + f'aicore_intermediate_{self.rank_id}_detail.csv'
        step_trace_file = self.profiler_path + f'step_trace_raw_{self.rank_id}_detail_time.csv'
        timeline_file = self.profiler_path + f'ascend_timeline_display_{self.rank_id}.json'
        aicpu_file = self.profiler_path + f'aicpu_intermediate_{self.rank_id}.csv'
        minddata_pipeline_file = self.profiler_path + f'minddata_pipeline_raw_{self.rank_id}.csv'
        queue_profiling_file = self.profiler_path + f'device_queue_profiling_{self.rank_id}.txt'
        memory_file = self.profiler_path + f'memory_usage_{self.rank_id}.pb'

        d_profiler_files = (aicore_file, step_trace_file, timeline_file, aicpu_file,
                            minddata_pipeline_file, queue_profiling_file, memory_file)
        for file in d_profiler_files:
            assert os.path.isfile(file)

    def _check_cpu_profiling_file(self):
        op_detail_file = self.profiler_path + f'cpu_op_detail_info_{self.device_id}.csv'
        op_type_file = self.profiler_path + f'cpu_op_type_info_{self.device_id}.csv'
        timeline_file = self.profiler_path + f'cpu_op_execute_timestamp_{self.device_id}.txt'

        cpu_profiler_files = (op_detail_file, op_type_file, timeline_file)
        for file in cpu_profiler_files:
            assert os.path.isfile(file)

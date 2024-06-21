# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
import tempfile
from collections import defaultdict
import json
import sys
import csv

from tests.security_utils import security_off_wrap
import pytest

import mindspore as ms
from mindspore import dataset as ds
from mindspore import nn, Tensor, context
from mindspore.nn.optim import Momentum
from mindspore.dataset.transforms import transforms as C
from mindspore.dataset.vision import transforms as CV
from mindspore.dataset.vision import Inter
from mindspore.common import dtype as mstype
from mindspore.common.initializer import TruncatedNormal
from mindspore.train import Model, Accuracy
from mindspore import Profiler


mnist_path = '/home/workspace/mindspore_dataset/mnist'


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
        self.conv1.conv2d.add_prim_attr("primitive_target", "CPU")
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


@pytest.mark.level3
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@security_off_wrap
def test_cpu_profiler():
    """
    Feature: profiler support cpu mode.
    Description: profiling op time and timeline.
    Expectation: No exception.
    """
    if sys.platform != 'linux':
        return
    device_id = 0
    data_path = tempfile.mkdtemp(prefix='profiler_data', dir='/tmp')
    profiler_path = os.path.join(data_path, 'profiler/')
    try:
        _train_with_profiler(data_path=data_path, device_target="CPU", profile_memory=False)
        _check_cpu_profiling_file(profiler_path, device_id)
    finally:
        if os.path.exists(data_path):
            shutil.rmtree(data_path)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@security_off_wrap
def test_gpu_profiler():
    """
    Feature: profiler support GPU  mode.
    Description: profiling op time and timeline.
    Expectation: No exception.
    """
    device_id = int(os.getenv('DEVICE_ID')) if os.getenv('DEVICE_ID') else 0
    rank_id = int(os.getenv('RANK_ID')) if os.getenv('RANK_ID') else 0
    data_path = tempfile.mkdtemp(prefix='profiler_data', dir='/tmp')
    profiler_path = os.path.join(data_path, 'profiler/')
    try:
        _train_with_profiler(data_path=data_path, device_target="GPU", profile_memory=False,
                             context_mode=context.GRAPH_MODE)
        _check_gpu_profiling_file(profiler_path, device_id)
        _check_host_profiling_file(profiler_path, rank_id)
    finally:
        if os.path.exists(data_path):
            shutil.rmtree(data_path)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@security_off_wrap
def test_gpu_profiler_pynative():
    """
    Feature: profiler support GPU pynative mode.
    Description: profiling l2 GPU pynative mode data, analyze performance issues.
    Expectation: No exception.
    """
    device_id = int(os.getenv('DEVICE_ID')) if os.getenv('DEVICE_ID') else 0
    rank_id = int(os.getenv('RANK_ID')) if os.getenv('RANK_ID') else 0
    data_path = tempfile.mkdtemp(prefix='profiler_data', dir='/tmp')
    profiler_path = os.path.join(data_path, 'profiler/')
    try:
        _train_with_profiler(data_path=data_path, device_target="GPU", profile_memory=False,
                             context_mode=context.PYNATIVE_MODE)
        _check_gpu_profiling_file(profiler_path, device_id)
        _check_host_profiling_file(profiler_path, rank_id)
    finally:
        if os.path.exists(data_path):
            shutil.rmtree(data_path)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_ascend_profiler():
    """
    Feature: profiler support ascend mode.
    Description: profiling op time, timeline, step trace and host data.
    Expectation: No exception.
    """
    ms.set_context(jit_level="O2")
    rank_id = int(os.getenv('RANK_ID')) if os.getenv('RANK_ID') else 0
    data_path = tempfile.mkdtemp(prefix='profiler_data', dir='/tmp')
    profiler_path = os.path.join(data_path, 'profiler/')
    try:
        _train_with_profiler(data_path=data_path, device_target="Ascend", profile_memory=True)
        _check_d_profiling_file(profiler_path, rank_id)
        _check_d_profiling_step_trace_on_multisubgraph(profiler_path, rank_id)
        _check_host_profiling_file(profiler_path, rank_id)
    finally:
        if os.path.exists(data_path):
            shutil.rmtree(data_path)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
@pytest.mark.parametrize("profile_framework", ['all', 'time', 'memory', None])
def test_host_profiler(profile_framework):
    """
    Feature: profiling support ascend kbyk mode.
    Description: profiling kbyk host data.
    Expectation: No exception.
    """
    rank_id = int(os.getenv('RANK_ID')) if os.getenv('RANK_ID') else 0
    data_path = tempfile.mkdtemp(prefix='profiler_data', dir='/tmp')
    profiler_path = os.path.join(data_path, 'profiler/')
    try:
        _train_with_profiler(data_path=data_path, device_target="Ascend", profile_memory=False, only_profile_host=True,
                             profile_framework=profile_framework)
        _check_host_profiling_file(profiler_path, rank_id, profile_framework=profile_framework)
    finally:
        if os.path.exists(data_path):
            shutil.rmtree(data_path)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_ascend_kbyk_profiler():
    """
    Feature: profiling ascend kbyk host data.
    Description: profiling ascend and host data.
    Expectation: No exception.
    """
    os.environ['GRAPH_OP_RUN'] = "1"
    rank_id = int(os.getenv('RANK_ID')) if os.getenv('RANK_ID') else 0
    data_path = tempfile.mkdtemp(prefix='profiler_data', dir='/tmp')
    profiler_path = os.path.join(data_path, 'profiler/')
    try:
        _train_with_profiler(data_path=data_path, device_target="Ascend", profile_memory=False, host_stack=True)
        _check_d_profiling_file(profiler_path, rank_id)
        _check_host_profiling_file(profiler_path, rank_id)
        _check_kbyk_profiling_file(profiler_path, rank_id)
        del os.environ['GRAPH_OP_RUN']
    finally:
        if os.path.exists(data_path):
            shutil.rmtree(data_path)


def _check_kbyk_profiling_file(profiler_path, rank_id):
    op_range_file = os.path.join(profiler_path, "FRAMEWORK/op_range_" + str(rank_id))
    assert os.path.isfile(op_range_file)


def _train_with_profiler(device_target, profile_memory, data_path, context_mode=context.GRAPH_MODE,
                         only_profile_host=False, profile_framework='all', host_stack=True):
    context.set_context(mode=context_mode, device_target=device_target)
    ds_train = create_dataset(os.path.join(mnist_path, "train"))
    if ds_train.get_dataset_size() == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")
    if only_profile_host:
        profiler = Profiler(output_path=data_path, op_time=False,
                            parallel_strategy=False, aicore_metrics=-1, data_process=False,
                            profile_framework=profile_framework, host_stack=host_stack, data_simplification=False)
    else:
        profiler = Profiler(profile_memory=profile_memory, output_path=data_path,
                            profile_framework=profile_framework, host_stack=host_stack, data_simplification=False)
    lenet = LeNet5()
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    optim = Momentum(lenet.trainable_params(), learning_rate=0.1, momentum=0.9)
    model = Model(lenet, loss_fn=loss, optimizer=optim, metrics={'acc': Accuracy()})

    model.train(1, ds_train, dataset_sink_mode=True)
    profiler.analyse()
    if device_target != 'Ascend':
        profiler.op_analyse(op_name="Conv2D")


def _check_gpu_profiling_file(profiler_path, device_id):
    op_detail_file = profiler_path + f'gpu_op_detail_info_{device_id}.csv'
    op_type_file = profiler_path + f'gpu_op_type_info_{device_id}.csv'
    activity_file = profiler_path + f'gpu_activity_data_{device_id}.csv'
    timeline_file = profiler_path + f'gpu_timeline_display_{device_id}.json'
    getnext_file = profiler_path + f'minddata_getnext_profiling_{device_id}.txt'
    pipeline_file = profiler_path + f'minddata_pipeline_raw_{device_id}.csv'
    framework_file = profiler_path + f'gpu_framework_{device_id}.txt'

    gpu_profiler_files = (op_detail_file, op_type_file, activity_file,
                          timeline_file, getnext_file, pipeline_file, framework_file)
    for file in gpu_profiler_files:
        assert os.path.isfile(file)


def _check_d_profiling_step_trace_on_multisubgraph(profiler_path, rank_id):
    step_trace_file = profiler_path + f'step_trace_raw_{rank_id}_detail_time.csv'
    assert os.path.isfile(step_trace_file)
    with open(step_trace_file, 'r') as fr:
        reader = csv.DictReader(fr)
        row_count = sum(1 for _ in reader)
    assert row_count == 11


def _check_d_profiling_file(profiler_path, rank_id):
    aicore_file = profiler_path + f'aicore_intermediate_{rank_id}_detail.csv'
    timeline_file = profiler_path + f'ascend_timeline_display_{rank_id}.json'
    aicpu_file = profiler_path + f'aicpu_intermediate_{rank_id}.csv'
    minddata_pipeline_file = profiler_path + f'minddata_pipeline_raw_{rank_id}.csv'
    queue_profiling_file = profiler_path + f'device_queue_profiling_{rank_id}.txt'

    d_profiler_files = (aicore_file, timeline_file, aicpu_file,
                        minddata_pipeline_file, queue_profiling_file)
    for file in d_profiler_files:
        assert os.path.isfile(file)


def _check_cpu_profiling_file(profiler_path, device_id):
    op_detail_file = profiler_path + f'cpu_op_detail_info_{device_id}.csv'
    op_type_file = profiler_path + f'cpu_op_type_info_{device_id}.csv'
    timeline_file = profiler_path + f'cpu_op_execute_timestamp_{device_id}.txt'

    cpu_profiler_files = (op_detail_file, op_type_file, timeline_file)
    for file in cpu_profiler_files:
        assert os.path.isfile(file)


def _check_host_profiling_file(profiler_path, rank_id, profile_framework='all'):
    host_dir = os.path.join(profiler_path, 'host_info')
    if profile_framework is None:
        assert not os.path.exists(host_dir)
        return
    if profile_framework in ['all', 'time']:
        timeline_file = os.path.join(host_dir, f'timeline_{rank_id}.json')
        assert os.path.isfile(timeline_file)
    csv_file = os.path.join(host_dir, f'host_info_{rank_id}.csv')
    assert os.path.exists(csv_file)
    with open(csv_file, 'r') as f:
        f_reader = csv.reader(f)
        header = next(f_reader)
        assert header == ['tid', 'pid', 'parent_pid', 'module_name', 'event', 'stage', 'level', 'start_end',
                          'custom_info', 'memory_usage(kB)', 'time_stamp(us)']
        for row in f_reader:
            assert len(row) == 11


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_ascend_pynative_profiler():
    """
    Feature: profiling ascend pynative host data.
    Description: profiling pynative host data.
    Expectation: No exception.
    """
    rank_id = int(os.getenv('RANK_ID')) if os.getenv('RANK_ID') else 0
    data_path = tempfile.mkdtemp(prefix='profiler_data', dir='/tmp')
    profiler_path = os.path.join(data_path, 'profiler/')
    try:
        _train_with_profiler(data_path=data_path, device_target='Ascend', profile_memory=False,
                             context_mode=context.PYNATIVE_MODE, host_stack=True)
        _check_pynative_timeline_host_data(profiler_path, rank_id)
    finally:
        if os.path.exists(data_path):
            shutil.rmtree(data_path)


def _check_pynative_timeline_host_data(profiler_path, rank_id):
    timeline_display_file = os.path.join(profiler_path, f'ascend_timeline_display_{rank_id}.json')
    assert os.path.isfile(timeline_display_file)
    with open(timeline_display_file, 'r') as fr:
        data = json.load(fr)
    async_ms_dict, async_npu_dict, host_to_device_dict = defaultdict(int), defaultdict(int), defaultdict(int)
    RunOp_set, FrontendTask_set, DeviceTask_set, LaunchTask_set, KernelLaunch_set \
        = set(), set(), set(), set(), set()

    for d in data:
        ph = d.get('ph')
        cat = d.get('cat')
        name = d.get('name')
        if ph in ('s', 'f'):
            if cat == 'async_mindspore':
                async_ms_dict[d.get('id')] += 1
            elif cat == 'async_npu':
                async_npu_dict[d.get('id')] += 1
            elif cat == 'HostToDevice':
                host_to_device_dict[d.get('id')] += 1
        elif ph == 'X':
            if 'RunOp' in name:
                assert d.get('args', {}).get('Call stack')
                RunOp_set.add(name)
            elif 'FrontendTask' in name:
                FrontendTask_set.add(name)
            elif 'DeviceTask' in name:
                DeviceTask_set.add(name)
            elif 'LaunchTask' in name:
                LaunchTask_set.add(name)
            elif 'KernelLaunch' in name:
                KernelLaunch_set.add(name)

    assert RunOp_set
    assert FrontendTask_set
    assert DeviceTask_set
    assert LaunchTask_set
    assert KernelLaunch_set
    for v in async_ms_dict.values():
        assert v == 2
    for v in async_npu_dict.values():
        assert v == 2
    for v in host_to_device_dict.values():
        assert v == 2

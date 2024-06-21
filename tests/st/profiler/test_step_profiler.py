# Copyright 2023 Huawei Technologies Co., Ltd
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
import csv
import numpy as np
from mindspore import nn
import mindspore as ms
import mindspore.dataset as ds
from tests.security_utils import security_off_wrap
import pytest


class StopAtStep(ms.Callback):
    """
    Start profiling base on step.

    Args:
        start_step (int): The start step number.
        stop_step (int): The stop step number.
    """
    def __init__(self, start_step, stop_step, data_path):
        super(StopAtStep, self).__init__()
        self.start_step = start_step
        self.stop_step = stop_step
        self.profiler = ms.Profiler(start_profile=False, output_path=data_path)

    def on_train_step_begin(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.start_step:
            self.profiler.start()

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.stop_step:
            self.profiler.stop()
            self.profiler.analyse()


class Net(nn.Cell):
    """The test net"""
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Dense(2, 2)

    def construct(self, x):
        return self.fc(x)


def generator():
    for _ in range(10):
        yield (np.ones([2, 2]).astype(np.float32), np.ones([2]).astype(np.int32))


def cleanup():
    kernel_meta_path = os.path.join(os.getcwd(), "kernel_data")
    cache_path = os.path.join(os.getcwd(), "__pycache__")
    if os.path.exists(kernel_meta_path):
        shutil.rmtree(kernel_meta_path)
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)


class TestProfiler:
    device_id = int(os.getenv('DEVICE_ID')) if os.getenv('DEVICE_ID') else 0
    rank_id = int(os.getenv('RANK_ID')) if os.getenv('RANK_ID') else 0

    def setup(self):
        """Run begin each test case start."""
        cleanup()
        self.data_path = tempfile.mkdtemp(prefix='profiler_data', dir='/tmp')

    def teardown(self):
        """Run after each test case end."""
        cleanup()
        if os.path.exists(self.data_path):
            shutil.rmtree(self.data_path)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    @security_off_wrap
    def test_ascend_profiler(self):
        ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
        ms.set_context(jit_level="O2")

        profile_call_back = StopAtStep(5, 8, self.data_path)

        net = Net()
        optimizer = nn.Momentum(net.trainable_params(), 1, 0.9)
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        data = ds.GeneratorDataset(generator, ["data", "label"])
        model = ms.Model(net, loss, optimizer)
        model.train(3, data, callbacks=[profile_call_back], dataset_sink_mode=False)
        profiler_name = 'profiler/'
        self.profiler_path = os.path.join(self.data_path, profiler_name)
        aicore_file = self.profiler_path + f'aicore_intermediate_{self.rank_id}_detail.csv'
        step_trace_file = self.profiler_path + f'step_trace_raw_{self.rank_id}_detail_time.csv'
        d_profiler_files = (aicore_file, step_trace_file)
        for file in d_profiler_files:
            assert os.path.isfile(file)
        with open(step_trace_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            row_count = sum(1 for row in reader)
            assert row_count == 6

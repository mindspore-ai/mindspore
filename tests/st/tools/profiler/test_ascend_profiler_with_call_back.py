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
"""test ascend profiler with call back."""
import glob
import os
import tempfile
import csv
import numpy as np
from mindspore import nn
import mindspore as ms
import mindspore.dataset as ds
from tests.mark_utils import arg_mark
from file_check import FileChecker

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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=1024.25)
def test_ascend_profiler():
    """
    Feature: Ascend Profiler
    Description: Test Ascend Profiler with step profiler.
    Expectation: The profiler successfully collects data and generates the expected files.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
        ms.set_context(jit_level="O2")

        profile_call_back = StopAtStep(5, 8, tmpdir)

        net = Net()
        optimizer = nn.Momentum(net.trainable_params(), 1, 0.9)
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        data = ds.GeneratorDataset(generator, ["data", "label"])
        model = ms.Model(net, loss, optimizer)
        model.train(1, data, callbacks=[profile_call_back], dataset_sink_mode=False)
        ascend_profiler_output_path = glob.glob(f"{tmpdir}/*_ascend_ms/ASCEND_PROFILER_OUTPUT")[0]
        aicore_file = os.path.join(ascend_profiler_output_path, f'kernel_details.csv')
        step_trace_file = os.path.join(ascend_profiler_output_path, f'step_trace_time.csv')
        d_profiler_files = (aicore_file, step_trace_file)
        for file in d_profiler_files:
            assert os.path.isfile(file)
        with open(step_trace_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            row_count = sum(1 for row in reader)
            assert row_count == 2
        # Check profiler.log
        profiler_log_paths = glob.glob(f"{tmpdir}/*_ascend_ms/"
                                       f"logs/profiler_*.log")
        for profiler_log_path in profiler_log_paths:
            FileChecker.check_file_for_keyword(profiler_log_path, "error")

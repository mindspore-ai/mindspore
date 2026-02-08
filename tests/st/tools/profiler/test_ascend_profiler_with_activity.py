# Copyright 2024-2025 Huawei Technologies Co., Ltd
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
"""test ascend profiler with activity"""
import tempfile
import glob
import numpy as np

import mindspore
import mindspore.dataset as ds
from mindspore.profiler import ProfilerActivity
from mindspore import context, nn
from mindspore.profiler.profiler_interface import ProfilerInterface

from file_check import FileChecker
from tests.mark_utils import arg_mark

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Dense(2, 2)

    def construct(self, x):
        return self.fc(x)


def generator_net():
    for _ in range(2):
        yield np.ones([2, 2]).astype(np.float32), np.ones([2]).astype(np.int32)


def train_net(net):
    optimizer = nn.Momentum(net.trainable_params(), 1, 0.9)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    data = ds.GeneratorDataset(generator_net(), ["data", "label"])
    model = mindspore.train.Model(net, loss, optimizer)
    model.train(1, data)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='unessential', mem_peak=2048.0)
def test_dynamic_step_npu_py_native_profiler():
    """
    Feature: Dynamic Step Profiler
    Description: This test case verifies that the profiler can correctly profile the network at CPU device.
    Expectation: The profiler should profile the network without any exceptions and
    generate the expected profiling data.
    """
    step_num = 8
    with tempfile.TemporaryDirectory(suffix="_step_profiler_npu") as tmpdir:
        schedule = mindspore.profiler.schedule(wait=1, warmup=1, active=1, repeat=1, skip_first=1)
        net = Net()
        context.set_context(mode=mindspore.PYNATIVE_MODE, device_target="Ascend")
        # pylint: disable=protected-access
        experimental_config = mindspore.profiler._ExperimentalConfig()
        profile = mindspore.profiler.profile(activities=[ProfilerActivity.NPU],
                                             schedule=schedule,
                                             on_trace_ready=mindspore.profiler.tensorboard_trace_handler(
                                                 dir_name=tmpdir),
                                             experimental_config=experimental_config)
        for _ in range(step_num):
            train_net(net)
            profile.step()
        ProfilerInterface.finalize()
        ProfilerInterface.clear()
        _check_npu_profiler_data(tmpdir)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=2048.0)
def test_dynamic_step_cpu_py_native_profiler():
    """
    Feature: Dynamic Step Profiler
    Description: This test case verifies that the profiler can correctly profile the network at CPU device.
    Expectation: The profiler should profile the network without any exceptions and
    generate the expected profiling data.
    """
    step_num = 8
    with tempfile.TemporaryDirectory(suffix="_step_profiler_cpu") as tmpdir:
        schedule = mindspore.profiler.schedule(wait=1, warmup=1, active=1, repeat=1, skip_first=1)
        net = Net()
        context.set_context(mode=mindspore.PYNATIVE_MODE, device_target="Ascend")
        # pylint: disable=protected-access
        experimental_config = mindspore.profiler._ExperimentalConfig()
        profile = mindspore.profiler.profile(activities=[ProfilerActivity.CPU],
                                             schedule=schedule,
                                             on_trace_ready=mindspore.profiler.tensorboard_trace_handler(
                                                 dir_name=tmpdir),
                                             experimental_config=experimental_config)
        for _ in range(step_num):
            train_net(net)
            profile.step()
        ProfilerInterface.finalize()
        ProfilerInterface.clear()
        _check_cpu_profiler_data(tmpdir)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
def test_dynamic_step_npu_graph_profiler():
    """
    Feature: Dynamic Step Profiler
    Description: This test case verifies that the profiler can correctly profile the network at CPU device.
    Expectation: The profiler should profile the network without any exceptions and
    generate the expected profiling data.
    """
    step_num = 8
    with tempfile.TemporaryDirectory(suffix="_step_profiler_npu") as tmpdir:
        schedule = mindspore.profiler.schedule(wait=1, warmup=1, active=1, repeat=1, skip_first=1)
        net = Net()
        context.set_context(mode=mindspore.GRAPH_MODE, device_target="Ascend")
        context.set_context(jit_config={"jit_level": "O0"})
        # pylint: disable=protected-access
        experimental_config = mindspore.profiler._ExperimentalConfig()
        profile = mindspore.profiler.profile(activities=[ProfilerActivity.NPU],
                                             schedule=schedule,
                                             on_trace_ready=mindspore.profiler.tensorboard_trace_handler(
                                                 dir_name=tmpdir),
                                             experimental_config=experimental_config)
        for _ in range(step_num):
            train_net(net)
            profile.step()
        ProfilerInterface.finalize()
        ProfilerInterface.clear()
        _check_npu_profiler_data(tmpdir)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
def test_dynamic_step_cpu_graph_profiler():
    """
    Feature: Dynamic Step Profiler
    Description: This test case verifies that the profiler can correctly profile the network at CPU device.
    Expectation: The profiler should profile the network without any exceptions and
    generate the expected profiling data.
    """
    step_num = 8
    with tempfile.TemporaryDirectory(suffix="_step_profiler_cpu") as tmpdir:
        schedule = mindspore.profiler.schedule(wait=1, warmup=1, active=1, repeat=1, skip_first=1)
        net = Net()
        context.set_context(mode=mindspore.GRAPH_MODE, device_target="Ascend")
        context.set_context(jit_config={"jit_level": "O0"})
        # pylint: disable=protected-access
        experimental_config = mindspore.profiler._ExperimentalConfig()
        profile = mindspore.profiler.profile(activities=[ProfilerActivity.CPU],
                                             schedule=schedule,
                                             on_trace_ready=mindspore.profiler.tensorboard_trace_handler(
                                                 dir_name=tmpdir),
                                             experimental_config=experimental_config)
        for _ in range(step_num):
            train_net(net)
            profile.step()
        ProfilerInterface.finalize()
        ProfilerInterface.clear()
        _check_cpu_profiler_data(tmpdir)

def _check_npu_profiler_data(tmpdir):
    """ Check only NPU profiler data."""
    # Check kernel_details.csv
    kernel_details_path = glob.glob(f"{tmpdir}/*_ascend_ms/"
                                    f"ASCEND_PROFILER_OUTPUT/kernel_details.csv")[0]
    FileChecker.assert_csv_no_header(kernel_details_path, "Step ID")
    FileChecker.check_csv_items(kernel_details_path, {"Name": ["*BiasAdd*", "*MatMul*"]})
    # Check trace_view.json
    trace_view_path = glob.glob(f"{tmpdir}/*_ascend_ms/"
                                f"ASCEND_PROFILER_OUTPUT/trace_view.json")[0]
    FileChecker.check_timeline_values(
        trace_view_path,
        "name",
        ["*MatMul*",
         "*Add*"
         ],
        fuzzy_match=True
    )
    # Check step_trace_time.csv
    step_trace_time_path = glob.glob(f"{tmpdir}/*_ascend_ms/"
                                     f"ASCEND_PROFILER_OUTPUT/step_trace_time.csv")[0]
    FileChecker.check_csv_data_non_negative(step_trace_time_path, comparison_func=_is_non_negative)
    # Check api_statistic.csv
    api_statistic_path = glob.glob(f"{tmpdir}/*_ascend_ms/"
                                   f"ASCEND_PROFILER_OUTPUT/api_statistic.csv")[0]
    FileChecker.check_file_exists(api_statistic_path)
    # Check profiler.log
    profiler_log_paths = glob.glob(f"{tmpdir}/*_ascend_ms/"
                                   f"logs/profiler_*.log")
    for profiler_log_path in profiler_log_paths:
        FileChecker.check_file_for_keyword(profiler_log_path, "error")


def _check_cpu_profiler_data(tmpdir):
    """ Check only CPU profiler data."""
    # Check trace_view.json
    trace_view_path = glob.glob(f"{tmpdir}/*_ascend_ms/"
                                f"ASCEND_PROFILER_OUTPUT/trace_view.json")[0]
    FileChecker.check_timeline_values(
        trace_view_path,
        "name",
        ["*ProfilerStep#3"
         ],
        fuzzy_match=True
    )
    # Check profiler.log
    profiler_log_paths = glob.glob(f"{tmpdir}/*_ascend_ms/"
                                   f"logs/profiler_*.log")
    for profiler_log_path in profiler_log_paths:
        FileChecker.check_file_for_keyword(profiler_log_path, "error")
    # Check dataset.csv
    dataset_path = glob.glob(f"{tmpdir}/*_ascend_ms/"
                             f"ASCEND_PROFILER_OUTPUT/dataset.csv")[0]
    FileChecker.check_file_exists(dataset_path)


def _is_non_negative(value):
    """ Check if a given value is non-negative (i.e., greater than or equal to zero)."""
    return float(value) >= 0.0

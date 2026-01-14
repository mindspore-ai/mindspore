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
"""test ascend profiler with start stop"""
import glob
import os
import tempfile
import numpy as np

from mindspore import Tensor, context
import mindspore as ms
from mindspore.profiler import Profiler
from mindspore.profiler import ProfilerLevel

from file_check import FileChecker
from model_zoo import TinyAddNet
from tests.mark_utils import arg_mark


def train(add):
    """ Train add net"""
    x = np.random.randn(1, 3, 3, 4).astype(np.float32)
    y = np.random.randn(1, 3, 3, 4).astype(np.float32)
    add(Tensor(x), Tensor(y))


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=0.01)
def test_dynamic_start_stop_kbk_profiler():
    """
    Feature: Dynamic Start-Stop Profiler
    Description: This test case verifies that the profiler can dynamically start and stop
    profiling the network at specified steps.
    Expectation: The profiler should start and stop profiling the network without any exceptions
    and generate the expected profiling data.
    """
    with tempfile.TemporaryDirectory(suffix="_start_stop_profiler") as tmpdir:
        add = TinyAddNet()
        _dynamic_start_stop_train_profiler(tmpdir, add, ms.GRAPH_MODE, "O0")
        # Check whether the number of generated files is the same as the data collected by the step
        ascend_ms_dir_nums = len(glob.glob(f"{tmpdir}/*_ascend_ms"))
        assert ascend_ms_dir_nums == 1
        # Check kernel_details.csv
        kernel_details_path_step = os.path.join(
            glob.glob(f"{tmpdir}/*_ascend_ms")[0],
            "ASCEND_PROFILER_OUTPUT",
            "kernel_details.csv"
        )
        FileChecker.assert_csv_no_header(kernel_details_path_step, "Step ID")
        # Check profiler.log
        profiler_log_paths = glob.glob(f"{tmpdir}/*_ascend_ms/"
                                       f"logs/profiler_*.log")
        for profiler_log_path in profiler_log_paths:
            FileChecker.check_file_for_keyword(profiler_log_path, "error")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1024.01)
def test_dynamic_start_stop_graph_profiler():
    """
    Feature: Dynamic Start-Stop Profiler
    Description: This test case verifies that the profiler can dynamically start and stop
    profiling the network at specified steps.
    Expectation: The profiler should start and stop profiling the network without any exceptions
    and generate the expected profiling data.
    """
    with tempfile.TemporaryDirectory(suffix="_start_stop_profiler") as tmpdir:
        add = TinyAddNet()
        _dynamic_start_stop_train_profiler(tmpdir, add, ms.GRAPH_MODE, "O2")
        # Check whether the number of generated files is the same as the data collected by the step
        ascend_ms_dir_nums = len(glob.glob(f"{tmpdir}/*_ascend_ms"))
        assert ascend_ms_dir_nums == 1
        # Check kernel_details.csv
        kernel_details_path_step = os.path.join(
            glob.glob(f"{tmpdir}/*_ascend_ms")[0],
            "ASCEND_PROFILER_OUTPUT",
            "kernel_details.csv"
        )
        FileChecker.assert_csv_no_header(kernel_details_path_step, "Step ID")
        # Check profiler.log
        profiler_log_paths = glob.glob(f"{tmpdir}/*_ascend_ms/"
                                       f"logs/profiler_*.log")
        for profiler_log_path in profiler_log_paths:
            FileChecker.check_file_for_keyword(profiler_log_path, "error")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1024.00)
def test_dynamic_start_stop_py_native_profiler():
    """
    Feature: Dynamic Start-Stop Profiler
    Description: This test case verifies that the profiler can dynamically start and stop
    profiling the network at specified steps.
    Expectation: The profiler should start and stop profiling the network without any exceptions
    and generate the expected profiling data.
    """
    with tempfile.TemporaryDirectory(suffix="_start_stop_profiler") as tmpdir:
        add = TinyAddNet()
        _dynamic_start_stop_train_profiler(tmpdir, add, ms.PYNATIVE_MODE)
        # Check whether the number of generated files is the same as the data collected by the step
        ascend_ms_dir_nums = len(glob.glob(f"{tmpdir}/*_ascend_ms"))
        assert ascend_ms_dir_nums == 1
        # Check kernel_details.csv
        kernel_details_path_step = os.path.join(
            glob.glob(f"{tmpdir}/*_ascend_ms")[0],
            "ASCEND_PROFILER_OUTPUT",
            "kernel_details.csv"
        )
        FileChecker.assert_csv_no_header(kernel_details_path_step, "Step ID")
        # Check profiler.log
        profiler_log_paths = glob.glob(f"{tmpdir}/*_ascend_ms/"
                                       f"logs/profiler_*.log")
        for profiler_log_path in profiler_log_paths:
            FileChecker.check_file_for_keyword(profiler_log_path, "error")


def _dynamic_start_stop_train_profiler(tmpdir, add, context_mode, jit_level=None):
    """ Collect performance data according to start to stop"""
    context.set_context(mode=context_mode, device_target="Ascend")
    if jit_level:
        context.set_context(jit_config={"jit_level": jit_level})
    profiler = Profiler(profiler_level=ProfilerLevel.Level2,
                        data_process=False,
                        profile_framework="time",
                        l2_cache=True,
                        output_path=tmpdir)
    train(add)
    profiler.analyse()

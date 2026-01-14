# Copyright 2024 Huawei Technologies Co., Ltd
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
import glob
import tempfile
import shutil
import mindspore
from mindspore import context
from mindspore import Profiler
from mindspore import Tensor
from mindspore.profiler import ExportType
from mindspore.profiler.profiler import analyse
from mindspore.profiler.profiler_interface import ProfilerInterface

from tests.mark_utils import arg_mark
from file_check import FileChecker
from model_zoo import TinyAddNet


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1024.01)
def test_ascend_profiler_offline_analyse_with_single_device():
    """
    Feature: Ascend Offline Profiler Analysis with Single Device
    Description: Execute offline analysis with the Ascend profiler in PyNative mode for a single device, using a simple
                 addition network.
    Expectation: The profiler successfully performs offline analysis, and the expected analysis files are generated in
                 the temporary directory.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    with tempfile.TemporaryDirectory() as tmpdir:
        profiler = Profiler(output_path=tmpdir)
        net = TinyAddNet()
        t0 = Tensor(dtype=mindspore.float32, shape=[32, 32])
        t1 = Tensor(dtype=mindspore.float32, shape=[32, 32])
        net(t0, t1)
        profiler.stop()
        ProfilerInterface.finalize()
        ProfilerInterface.clear()
        profiler.offline_analyse(path=tmpdir, data_simplification=False)
        ascend_ms_dir = glob.glob(f"{tmpdir}/*_ascend_ms")[0]
        check_ascend_offline_analyse_files(ascend_ms_dir)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1024.00)
def test_ascend_profiler_offline_analyse_with_multi_devices():
    """
    Feature: Ascend Offline Profiler Analysis with Multi Devices
    Description: Test the Ascend profiler offline analysis capability in PyNative mode with multiple devices, using
                 a simple addition network.
    Expectation: Profiler offline analysis should successfully process data from multiple devices, with results
                 properly generated in the corresponding rank directories.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    with tempfile.TemporaryDirectory() as tmpdir:
        profiler = Profiler(output_path=tmpdir)
        net = TinyAddNet()
        t0 = Tensor(dtype=mindspore.float32, shape=[32, 32])
        t1 = Tensor(dtype=mindspore.float32, shape=[32, 32])
        net(t0, t1)
        profiler.stop()
        ProfilerInterface.finalize()
        ProfilerInterface.clear()
        # copy profiler data to rank0 and rank1
        raw_ascend_ms_dir = glob.glob(f"{tmpdir}/*_ascend_ms")[0]
        copy_ascend_ms_dir = os.path.join(tmpdir, 'copy_ascend_ms')
        shutil.copytree(raw_ascend_ms_dir, copy_ascend_ms_dir)
        profiler.offline_analyse(path=tmpdir, data_simplification=False)
        check_ascend_offline_analyse_files(raw_ascend_ms_dir)
        check_ascend_offline_analyse_files(copy_ascend_ms_dir)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.01)
def test_new_profiler_offline_analyse_with_single_device():
    """
    Feature: Ascend Offline Profiler Analysis with Single Device
    Description: Execute offline analysis with the Ascend profiler in PyNative mode for a single device, using a simple
                 addition network.
    Expectation: The profiler successfully performs offline analysis, and the expected analysis files are generated in
                 the temporary directory.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    with tempfile.TemporaryDirectory() as tmpdir:
        # pylint: disable=W0212
        experimental_config = mindspore.profiler._ExperimentalConfig(export_type=[ExportType.Text, ExportType.Db])
        profile = mindspore.profiler.profile(experimental_config=experimental_config,
                                             on_trace_ready=mindspore.profiler.tensorboard_trace_handler(
                                                 tmpdir, analyse_flag=False))
        net = TinyAddNet()
        t0 = Tensor(dtype=mindspore.float32, shape=[32, 32])
        t1 = Tensor(dtype=mindspore.float32, shape=[32, 32])
        net(t0, t1)
        profile.stop()
        analyse(profiler_path=tmpdir, data_simplification=False)
        ascend_ms_dir = glob.glob(f"{tmpdir}/*_ascend_ms")[0]
        check_ascend_offline_analyse_files(ascend_ms_dir, True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.01)
def test_new_profiler_offline_analyse_with_multi_devices():
    """
    Feature: Ascend Offline Profiler Analysis with Multi Devices
    Description: Test the Ascend profiler offline analysis capability in PyNative mode with multiple devices, using
                 a simple addition network.
    Expectation: Profiler offline analysis should successfully process data from multiple devices, with results
                 properly generated in the corresponding rank directories.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    with tempfile.TemporaryDirectory() as tmpdir:
        # pylint: disable=W0212
        experimental_config = mindspore.profiler._ExperimentalConfig(export_type=[ExportType.Text, ExportType.Db])
        profile = mindspore.profiler.profile(experimental_config=experimental_config,
                                             on_trace_ready=mindspore.profiler.tensorboard_trace_handler(
                                                 tmpdir, analyse_flag=False))
        net = TinyAddNet()
        t0 = Tensor(dtype=mindspore.float32, shape=[32, 32])
        t1 = Tensor(dtype=mindspore.float32, shape=[32, 32])
        net(t0, t1)
        profile.stop()
        # copy profiler data to rank0 and rank1
        raw_ascend_ms_dir = glob.glob(f"{tmpdir}/*_ascend_ms")[0]
        copy_ascend_ms_dir = os.path.join(tmpdir, 'copy_ascend_ms')
        shutil.copytree(raw_ascend_ms_dir, copy_ascend_ms_dir)
        analyse(profiler_path=tmpdir, data_simplification=False)
        check_ascend_offline_analyse_files(raw_ascend_ms_dir, True)
        check_ascend_offline_analyse_files(copy_ascend_ms_dir, True)


def check_ascend_offline_analyse_files(ascend_ms_dir: str, with_db: bool = False):
    """
    Check the Ascend offline analysis files.
    """
    ascend_profiler_output_path = os.path.join(ascend_ms_dir, "ASCEND_PROFILER_OUTPUT")

    if with_db:
        db_files = glob.glob(os.path.join(ascend_profiler_output_path, 'ascend_mindspore_profiler*.db'))
        assert len(db_files) >= 1
        FileChecker.check_file_exists(db_files[0])
    # Check trace_view.json
    trace_view_path = os.path.join(ascend_profiler_output_path, "trace_view.json")
    FileChecker.check_timeline_values(
        trace_view_path,
        "name",
        [
            "AscendCL@*",     # check CANN trace
            "HostToDevice*",  # check HostToDevice flow
        ],
        fuzzy_match=True
    )

    # check kernel_details.csv
    kernel_details_path = os.path.join(ascend_profiler_output_path, f"kernel_details.csv")
    FileChecker.check_csv_items(kernel_details_path, {"Name": ["*Add*"]})

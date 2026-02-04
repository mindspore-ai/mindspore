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
"""test ascend profiler with dynamic shape."""
import glob
import tempfile
import numpy as np
from tests.mark_utils import arg_mark
from model_zoo import DynamicShapeNet
from file_check import FileChecker

import mindspore
import mindspore.context as context
import mindspore.dataset as ds
from mindspore import Model
from mindspore import Profiler
from mindspore import Tensor
from mindspore.profiler import ProfilerLevel


def dataset_generator():
    # train 3 step
    for i in range(1, 3):
        yield np.ones((32, 2 * i), dtype=np.float32), np.ones((32, 2 * i), dtype=np.float32)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=1024.01)
def test_dynamic_shape_kbk_with_profiler_all_parameters_on():
    """
    Feature: Dynamic Shape
    Description: Test the Ascend profiler in pynative mode with Profiler, using a dynamic shape
                 model.
    Expectation: The profiler collects and analyzes data successfully, and the output files are correctly generated
                 in the temporary directory.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_level="O0")
    with tempfile.TemporaryDirectory() as tmpdir:
        network = DynamicShapeNet()
        profiler = Profiler(
            profiler_level=ProfilerLevel.Level1,
            output_path=tmpdir,
            op_time=True,
            profile_communication=True,
            parallel_strategy=True,
            start_profile=True,
            aicore_metrics=1,
            hbm_ddr=True,
            pcie=True,
            data_process=True,
            profile_framework="all",
            with_stack=True,
            data_simplification=False
        )

        dataset = ds.GeneratorDataset(dataset_generator, ["data1", "data2"])
        t0 = Tensor(dtype=mindspore.float32, shape=[32, None])
        t1 = Tensor(dtype=mindspore.float32, shape=[32, None])
        network.set_inputs(t0, t1)
        model = Model(network)
        model.train(1, dataset, dataset_sink_mode=True)
        profiler.analyse()

        # Check trace_view.json
        trace_view_path = glob.glob(f"{tmpdir}/*_ascend_ms/ASCEND_PROFILER_OUTPUT/trace_view.json")[0]
        FileChecker.check_timeline_values(
            trace_view_path,
            "name",
            [
                "mindspore/common/*",  # check stack trace
                "Dataset::*",  # check dataset trace
                "PynativeFramework::*",  # check host trace
                "AscendCL@*",  # check CANN trace
                "network-DynamicShapeNet",  # check scope layer
                "GetNext*",  # check kernel on Ascend Hardware
                "Free",  # check overlay analysis
                "HostToDevice*",  # check HostToDevice flow
                "mindspore_to_npu"  # check async_npu
            ],
            fuzzy_match=True
        )

        # check op_statistic.csv
        op_statistic_path = glob.glob(f"{tmpdir}/*_ascend_ms/ASCEND_PROFILER_OUTPUT/op_statistic.csv")[0]
        FileChecker.check_csv_items(op_statistic_path, {"OP Type": ["GetNext", "InitData", "Add"]})

        # check step_trace_time.csv
        step_trace_time_path = glob.glob(f"{tmpdir}/*_ascend_ms/"
                                         f"ASCEND_PROFILER_OUTPUT/step_trace_time.csv")[0]
        FileChecker.check_file_line_count(step_trace_time_path, 2)

        # check kernel_details.csv
        kernel_details_path = glob.glob(f"{tmpdir}/*_ascend_ms/"
                                        f"ASCEND_PROFILER_OUTPUT/kernel_details.csv")[0]
        FileChecker.check_csv_items(kernel_details_path, {"Name": ["*Add*", "GetNext*"]})
        # Check profiler.log
        profiler_log_paths = glob.glob(f"{tmpdir}/*_ascend_ms/"
                                       f"logs/profiler_*.log")
        for profiler_log_path in profiler_log_paths:
            FileChecker.check_file_for_keyword(profiler_log_path, "error")

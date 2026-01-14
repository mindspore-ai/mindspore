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
"""test ascend profiler with aicpu."""
import os.path
import tempfile
import numpy as np
import glob

import mindspore.context as context
import mindspore.common.dtype as mstype
from mindspore import Profiler
from mindspore import Tensor
from mindspore.profiler import ProfilerLevel
from tests.mark_utils import arg_mark
from model_zoo import CustomAICpuNet
from file_check import FileChecker


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=1024.01)
def test_collect_custom_aicpu():
    """
    Feature: Profiling can collect custom aicpu operators
    Description: Test profiling can collect custom aicpu operators on ascend
    Expectation: The file aicpu_intermediate_*.csv generated successfully and s1 == s2
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_level="O2")
    with tempfile.TemporaryDirectory(suffix="profiler_ai_cpu") as tmpdir:
        profiler = Profiler(output_path=tmpdir, profiler_level=ProfilerLevel.Level1)
        net = CustomAICpuNet()
        net(Tensor(np.random.random((6,)), mstype.float64))
        profiler.analyse()
        # Check op_statistic.csv
        op_dict = {"OP Type": ["Cast", "Select", "Xlogy"]}
        ascend_profiler_output_path = glob.glob(f"{tmpdir}/*_ascend_ms/ASCEND_PROFILER_OUTPUT")[0]
        FileChecker.check_csv_items(os.path.join(ascend_profiler_output_path, "op_statistic.csv"),
                                    op_dict, fuzzy_match=False)
        # Check profiler.log
        profiler_log_paths = glob.glob(f"{tmpdir}/*_ascend_ms/"
                                       f"logs/profiler_*.log")
        for profiler_log_path in profiler_log_paths:
            FileChecker.check_file_for_keyword(profiler_log_path, "error")

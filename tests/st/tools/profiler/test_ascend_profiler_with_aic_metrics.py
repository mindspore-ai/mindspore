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
"""test ascend profiler with aic metrics"""
import glob
import tempfile

import numpy as np

from mindspore import Tensor, context
import mindspore
from mindspore.profiler import ProfilerLevel, AicoreMetrics, ProfilerActivity

from file_check import FileChecker
from model_zoo import TinyAddNet
from tests.mark_utils import arg_mark


def train(add):
    """ Train add net"""
    x = np.random.randn(1, 3, 3, 4).astype(np.float32)
    y = np.random.randn(1, 3, 3, 4).astype(np.float32)
    add(Tensor(x), Tensor(y))


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
def test_aic_metrics_memory_access_kbk_profiler():
    """
    Feature: Aicore metrics is memory access Profiler
    Description: This test case verifies that the profiler can gather memory access data
    profiling the network at specified steps.
    Expectation: The profiler should step profiling the network without any exceptions
    and generate the expected profiling data.
    """
    with tempfile.TemporaryDirectory(suffix="_kbk_memory_access_profiler") as tmpdir:
        context.set_context(mode=mindspore.GRAPH_MODE, device_target="Ascend")
        context.set_context(jit_config={"jit_level": "O2"})
        add = TinyAddNet()
        # pylint: disable=protected-access
        experimental_config = mindspore.profiler._ExperimentalConfig(
            profiler_level=ProfilerLevel.Level1,
            aic_metrics=AicoreMetrics.MemoryAccess
        )
        with mindspore.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
                                        schedule=mindspore.profiler.schedule(wait=1, warmup=1, active=2,
                                                                             repeat=1, skip_first=1),
                                        on_trace_ready=mindspore.profiler.tensorboard_trace_handler(tmpdir),
                                        experimental_config=experimental_config) as prof:
            for _ in range(8):
                train(add)
                prof.step()
        # Check kernel_details.csv
        kernel_details_path = glob.glob(f"{tmpdir}/*_ascend_ms/"
                                        f"ASCEND_PROFILER_OUTPUT/kernel_details.csv")[0]
        FileChecker.check_csv_headers(kernel_details_path, ["aic_read_main_memory_datas(KB)",
                                                            "aic_write_main_memory_datas(KB)",
                                                            "aic_GM_to_L1_datas(KB)",
                                                            "aic_L0C_to_L1_datas(KB)",
                                                            "aic_L0C_to_GM_datas(KB)",
                                                            "aic_GM_to_UB_datas(KB)",
                                                            "aic_UB_to_GM_datas(KB)",
                                                            "aiv_read_main_memory_datas(KB)",
                                                            "aiv_write_main_memory_datas(KB)",
                                                            "aiv_GM_to_L1_datas(KB)",
                                                            "aiv_L0C_to_L1_datas(KB)",
                                                            "aiv_L0C_to_GM_datas(KB)",
                                                            "aiv_GM_to_UB_datas(KB)",
                                                            "aiv_UB_to_GM_datas(KB)"])
        # Check profiler.log
        profiler_log_paths = glob.glob(f"{tmpdir}/*_ascend_ms/"
                                       f"logs/profiler_*.log")
        for profiler_log_path in profiler_log_paths:
            FileChecker.check_file_for_keyword(profiler_log_path, "error")

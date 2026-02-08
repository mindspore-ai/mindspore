# Copyright 2025 Huawei Technologies Co., Ltd
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
"""profiler with ms tx."""
import os
import glob
import tempfile
import numpy as np

import mindspore as ms
from mindspore import context, Tensor
from mindspore.profiler import Profiler
from mindspore.profiler import ProfilerLevel, ProfilerActivity, AicoreMetrics, mstx
from tests.mark_utils import arg_mark
from tests.st.tools.profiler.file_check import FileChecker


def train(add):
    """ Train add net"""
    x = np.random.randn(1, 3, 3, 4).astype(np.float32)
    y = np.random.randn(1, 3, 3, 4).astype(np.float32)
    add(Tensor(x), Tensor(y))


class TinyNet(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.add = ms.ops.Add()

    def construct(self, x_, y_):
        stream = ms.runtime.current_stream()
        range_id1 = mstx.range_start("Add_net", stream=stream)
        mstx.mark("Start_add", stream=stream)
        out = self.add(x_, y_)
        mstx.mark("End_add", stream=stream)
        mstx.range_end(range_id1)
        return out


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_func_profiler_mstx_dot_with_schedule_001():
    """
    Feature: Profiler MSTX Marks
    Description: Test MSTX range/marks with schedule in PyNative.
    Expectation: Timeline contains MSTX range and mark events.
    """
    with tempfile.TemporaryDirectory(suffix="_mstx_func_profiler") as tmpdir:
        context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
        net = TinyNet()
        profiler_dir = os.path.join(tmpdir, "data")
        schedule = ms.profiler.schedule(wait=0, warmup=1, active=2, repeat=1, skip_first=1)
        prof = Profiler(profiler_level=ProfilerLevel.LevelNone,
                        activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
                        aicore_metrics=AicoreMetrics.PipeUtilization, with_stack=False,
                        profile_memory=False, data_process=False,
                        parallel_strategy=False, start_profile=True,
                        l2_cache=False, hbm_ddr=False, pcie=False,
                        sync_enable=True, data_simplification=True, mstx=True,
                        on_trace_ready=ms.profiler.tensorboard_trace_handler(dir_name=profiler_dir), schedule=schedule)
        prof.start()
        for _ in range(10):
            train(net)
            prof.step()
        prof.analyse()
        trace_view_json_path = glob.glob(os.path.join(profiler_dir, "*", "ASCEND_PROFILER_OUTPUT", "trace_view.json"))[
            0]
        FileChecker.check_timeline_values(
            trace_view_json_path,
            "name",
            [
                "Add_net"
            ],
        )
        FileChecker.check_timeline_values(
            trace_view_json_path,
            "name",
            [
                "Start_add"
            ],
        )

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
"""offline_analyse"""
import tempfile

import mindspore
from mindspore import context
from mindspore.profiler import ProfilerLevel, AicoreMetrics, ExportType
from mindspore.profiler.profiler import analyse

from tests.mark_utils import arg_mark
from tests.st.tools.profiler.model_zoo import TinyTransformer
from tests.st.tools.profiler.fake_dataset import FakeDataset
from tests.st.tools.profiler.daily_test.profiler_check import MSProfilerChecker


def collect_profiler_data():
    """
    Collect profiler data.
    """
    net = TinyTransformer(d_model=2, nhead=1, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=4)
    nlp_dataset = FakeDataset.create_fake_nlp_dataset(seq_len=1, batch_size=1, d_model=2, tgt_len=1, num_samples=1)
    for src, tgt in nlp_dataset:
        net(src, tgt)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_profiler_offline_analyse_path_correct_001():
    """
    Feature: Profiler Offline Analysis
    Description: Test profiler offline analysis with data simplification.
    Expectation: Generate analysis results with memory profiling and L2 cache metrics.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
        # pylint: disable=W0212
        experimental_config = mindspore.profiler._ExperimentalConfig(
            profiler_level=ProfilerLevel.Level2, aic_metrics=AicoreMetrics.PipeUtilization, l2_cache=True,
            data_simplification=True, export_type=[ExportType.Text])
        profiler = mindspore.profiler.profile(profile_memory=True, with_stack=True,
                                              experimental_config=experimental_config,
                                              on_trace_ready=mindspore.profiler.tensorboard_trace_handler(
                                                  dir_name=tmpdir))
        profiler.start()
        collect_profiler_data()
        profiler.stop()
        analyse(profiler_path=tmpdir, data_simplification=True)
        prof_config = {"output_path": tmpdir, "profiler_memory": True, "profiler_level": 2, "l2_cache": True}
        prof_check = MSProfilerChecker(prof_config, 1)
        prof_check()

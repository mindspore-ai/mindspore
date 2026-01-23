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
"""profiler with experimental config aic metrics."""
import tempfile

import mindspore as ms
from mindspore.profiler import ProfilerLevel, AicoreMetrics

from tests.mark_utils import arg_mark
from tests.st.tools.profiler.model_zoo import TinyTransformer
from tests.st.tools.profiler.fake_dataset import FakeDataset
from tests.st.tools.profiler.daily_test.profiler_check import MSProfilerChecker


def collect_profiler_data(tmpdir):
    """
    Collect profiler data.
    """
    # pylint: disable=W0212
    experimental_config = ms.profiler._ExperimentalConfig(
        profiler_level=ProfilerLevel.Level2,
        aic_metrics=AicoreMetrics.MemoryAccess,
    )
    schedule = ms.profiler.schedule(wait=0, warmup=0, active=4, repeat=1, skip_first=0)
    profile = ms.profiler.profile(schedule=schedule,
                                  on_trace_ready=ms.profiler.tensorboard_trace_handler(dir_name=tmpdir),
                                  experimental_config=experimental_config)
    net = TinyTransformer(d_model=2, nhead=1, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=4)
    nlp_dataset = FakeDataset.create_fake_nlp_dataset(seq_len=1, batch_size=1, d_model=2, tgt_len=1, num_samples=4)
    profile.start()
    for src, tgt in nlp_dataset:
        net(src, tgt)
        profile.step()
    profile.stop()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_func_profiler_resnet50_aic_metrics_memory_access_l2():
    """
    Feature: Profiler AIC Metrics
    Description: Test profiler with AIC memory access metrics at Level2.
    Expectation: Generate profiling data with experimental AIC metrics configuration.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        collect_profiler_data(tmpdir)
        prof_config = {"output_path": tmpdir, "pynative_step": True}
        prof_check = MSProfilerChecker(prof_config, 1, check_step_id=[0, 1, 2, 3])
        prof_check()

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
"""legacy single cards feature."""
import tempfile
from mindspore import Profiler
from mindspore.profiler import ProfilerLevel, ProfilerActivity, AicoreMetrics
import mindspore.profiler as Prof

from tests.mark_utils import arg_mark
from tests.st.tools.profiler.model_zoo import TinyTransformer
from tests.st.tools.profiler.fake_dataset import FakeDataset
from tests.st.tools.profiler.daily_test.profiler_check import MSProfilerChecker


class Config:
    def __init__(self, output_path):
        self.profiler_dir = output_path
        self.profiler_level = ProfilerLevel.Level2
        self.activities = [ProfilerActivity.CPU, ProfilerActivity.NPU]
        self.aicore_metrics = AicoreMetrics.PipeUtilization
        self.with_stack = True
        self.profile_memory = False
        self.data_process = True
        self.parallel_strategy = True
        self.start_profile = True
        self.l2_cache = True
        self.hbm_ddr = True
        self.pcie = True
        self.sync_enable = True
        self.data_simplification = False
        self.mstx = False


def generator_profiler_data(tmpdir, cfg):
    """
    Collect profiler data.
    """
    # Create Profiler instance with all parameters
    profiler = Profiler(
        output_path=cfg.profiler_dir,
        profiler_level=cfg.profiler_level,
        activities=cfg.activities,
        aic_metrics=cfg.aicore_metrics,
        with_stack=cfg.with_stack,
        profile_memory=cfg.profile_memory,
        data_process=cfg.data_process,
        parallel_strategy=cfg.parallel_strategy,
        start_profile=cfg.start_profile,
        l2_cache=cfg.l2_cache,
        hbm_ddr=cfg.hbm_ddr,
        pcie=cfg.pcie,
        sync_enable=cfg.sync_enable,
        data_simplification=cfg.data_simplification,
        mstx=cfg.mstx,
        on_trace_ready=Prof.tensorboard_trace_handler()
    )
    profiler.start()
    net = TinyTransformer(d_model=2, nhead=1, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=4)
    nlp_dataset = FakeDataset.create_fake_nlp_dataset(seq_len=1, batch_size=1, d_model=2, tgt_len=1, num_samples=5)
    for src, tgt in nlp_dataset:
        net(src, tgt)
    profiler.stop()
    profiler.analyse()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_func_profiler_single_card_resnet_sink_false_framework_all_14():
    """
    Feature: Legacy Profiler Single Card
    Description: Test single card ResNet profiling with sink_mode=False at Level0.
    Expectation: Generate profiling data with L2 cache and minddata metrics.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = Config(tmpdir)
        cfg.profile_memory = False
        cfg.profiler_level = ProfilerLevel.Level0
        generator_profiler_data(tmpdir, cfg)
        prof_config = {"output_path": cfg.profiler_dir,
                       "profile_memory": False,
                       "profile_level": 0,
                       "l2_cache": True,
                       "minddata": True}
        profiler_check = MSProfilerChecker(prof_config, 1)
        profiler_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_func_profiler_pynative_host_timeline_set_host_stack_true():
    """
    Feature: Profiler PyNative Host Timeline
    Description: Test PyNative mode profiling with host timeline and stack enabled.
    Expectation: Generate profiling data with memory, L2 cache, HBM DDR, and PCIe metrics.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = Config(tmpdir)
        cfg.with_stack = True
        cfg.profile_memory = True
        generator_profiler_data(tmpdir, cfg)
        prof_config = {"output_path": cfg.profiler_dir,
                       "profile_memory": True,
                       "profile_level": 2,
                       "l2_cache": True,
                       "hbm_ddr": True,
                       "pcie": True}
        profiler_check = MSProfilerChecker(prof_config, 1)
        profiler_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_func_profiler_single_card_resnet_sink_false_framework_memory_16():
    """
    Feature: Profiler Single Card Memory
    Description: Test single card ResNet profiling with memory UB metrics.
    Expectation: Generate profiling data with memory, L2 cache, and minddata metrics.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = Config(tmpdir)
        cfg.aicore_metrics = AicoreMetrics.MemoryUB
        cfg.profile_memory = True
        generator_profiler_data(tmpdir, cfg)
        prof_config = {"output_path": cfg.profiler_dir,
                       "profile_memory": True,
                       "profile_level": 2,
                       "l2_cache": True,
                       "minddata": True}
        profiler_check = MSProfilerChecker(prof_config, 1)
        profiler_check()

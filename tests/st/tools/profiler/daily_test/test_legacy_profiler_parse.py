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
"""legacy parse"""

import os
import glob
import json
import tempfile
from mindspore import context
from mindspore import Profiler

from tests.mark_utils import arg_mark
from tests.st.tools.profiler.model_zoo import TinyTransformer
from tests.st.tools.profiler.fake_dataset import FakeDataset


def collect_profiler_data(tmpdir, pretty_on=False):
    """Collect profiler data."""
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    profiler = Profiler(output_path=tmpdir)
    net = TinyTransformer(d_model=2, nhead=1, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=4)
    nlp_dataset = FakeDataset.create_fake_nlp_dataset(seq_len=1, batch_size=1, d_model=2, tgt_len=1, num_samples=1)
    for src, tgt in nlp_dataset:
        net(src, tgt)
    profiler.analyse(pretty=pretty_on)
    json_files = glob.glob(os.path.join(tmpdir, "*.json"))
    for json_file in json_files:
        with open(json_file, "r", encoding='utf-8') as f:
            data = json.load(f)
            assert data["profiler_parameters"]["profile_memory"] is False
            assert data["profiler_parameters"]["l2_cache"] is False


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_profiler_analyse_pretty_true_010():
    """
    Feature: Profiler Analyse with Pretty Print Enabled
    Description: Test the profiler analyse method with pretty_on=True to enable formatted JSON output.
    Expectation: The profiler successfully generates formatted JSON files in the output directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        collect_profiler_data(tmpdir, pretty_on=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_profiler_analyse_pretty_true_011():
    """
    Feature: Profiler Analyse with Pretty Print Disabled
    Description: Test the profiler analyse method with pretty_on=False to generate compressed JSON output.
    Expectation: The profiler successfully generates compressed JSON files in the output directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        collect_profiler_data(tmpdir, pretty_on=False)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_profiler_analyse_pretty_true_012():
    """
    Feature: Profiler Analyse with Non-Boolean Pretty Parameter
    Description: Test the profiler analyse method with pretty_on="123" (non-boolean value) to verify warning handling.
    Expectation: The profiler logs a warning for the invalid parameter and defaults to compressed JSON output.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        collect_profiler_data(tmpdir, pretty_on="123")

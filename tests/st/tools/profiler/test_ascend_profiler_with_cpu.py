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
"""test ascend profiler with cpu."""
import glob
import tempfile
from mindspore import context, Model, nn
from mindspore import Profiler
from tests.mark_utils import arg_mark
from file_check import FileChecker
from model_zoo import LeNet5
from fake_dataset import FakeDataset


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=0.01)
def test_cpu_profiler():
    """
    Feature: CPU Profiler
    Description: Test the CPU profiler.
    Expectation: The CPU profiler should collect and analyze data successfully.
    """
    with tempfile.TemporaryDirectory(suffix="_cpu_profiler") as tmpdir:
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
        profiler = Profiler(output_path=tmpdir)
        dataloader = FakeDataset.create_fake_cv_dataset()
        net = LeNet5()
        optimizer = nn.Momentum(net.trainable_params(), 0.01, 0.9)
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        model = Model(net, loss, optimizer)
        model.train(1, dataloader, dataset_sink_mode=False)
        profiler.analyse()

        # Check trace_view.json
        trace_view_path = glob.glob(f"{tmpdir}/*_ascend_ms/ASCEND_PROFILER_OUTPUT/trace_view.json")[0]
        FileChecker.check_timeline_values(
            trace_view_path,
            "name",
            [
                "*Conv2D*",
                "*BiasAdd*",
            ],
            fuzzy_match=True
        )
        # Check profiler.log
        profiler_log_paths = glob.glob(f"{tmpdir}/*_ascend_ms/"
                                       f"logs/profiler_*.log")
        for profiler_log_path in profiler_log_paths:
            FileChecker.check_file_for_keyword(profiler_log_path, "error")

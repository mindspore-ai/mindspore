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
"""profiler with model preparation."""
import os
import glob
import json
import tempfile

import mindspore as ms
from mindspore import context
from mindspore.profiler import profile, _ExperimentalConfig, ProfilerLevel, ProfilerActivity, ExportType, \
    tensorboard_trace_handler
from mindspore.train import Callback, Model
from mindspore import nn
from mindspore.nn.optim.momentum import Momentum
from mindspore.parallel._utils import _mstx_range_decorator

from tests.mark_utils import arg_mark
from tests.st.tools.profiler.daily_test.resnet50 import create_resnet50, create_dataset
from tests.st.tools.profiler.model_zoo import TinyTransformer
from tests.st.tools.profiler.fake_dataset import FakeDataset


class StopCallback(Callback):
    def __init__(self, stop_step):
        super().__init__()
        self.stop_step = stop_step

    def step_end(self, run_context):
        """Stop training at the specified step."""
        cb_params = run_context.original_args()
        if cb_params.cur_step_num == self.stop_step:
            run_context.request_stop()


def init_profiler(tmpdir):
    """init profiler."""
    experimental_config = _ExperimentalConfig(profiler_level=ProfilerLevel.LevelNone,
                                              export_type=[ExportType.Text, ExportType.Db], mstx=True,
                                              mstx_domain_include=["model_preparation"])
    prof = profile(start_profile=False, activities=[ProfilerActivity.NPU], experimental_config=experimental_config,
                   on_trace_ready=tensorboard_trace_handler(dir_name=tmpdir))
    return prof


def collect_profiler_data(tmpdir):
    """Collect profiler data."""
    prof = init_profiler(tmpdir)
    prof.start()
    net = TinyTransformer(d_model=2, nhead=1, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=4)
    nlp_dataset = FakeDataset.create_fake_nlp_dataset(seq_len=1, batch_size=1, d_model=2, tgt_len=1, num_samples=1)
    for src, tgt in nlp_dataset:
        net(src, tgt)
    prof.stop()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_func_profiler_with_model_preparation():
    """
    Feature: Profiler with model preparation
    Description: Test the profiler's ability to capture model preparation events in PyNative mode.
    Expectation: model_preparation event is captured in trace_view.json.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", jit_level="O0")
        collect_profiler_data(tmpdir)
        trace_view_path = glob.glob(os.path.join(tmpdir, "*", "ASCEND_PROFILER_OUTPUT", "trace_view.json"))[0]
        with open(trace_view_path, 'r', encoding='utf-8') as f:
            trace_data = json.load(f)
        trace_content = json.dumps(trace_data)
        assert "model_preparation" in trace_content, "model_preparation not found in trace_view.json"


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2918.0)
def test_func_profiler_with_resnet_train():
    """
    Feature: Profiler with resnet train.
    Description: Test the profiler's ability to capture model preparation events.
    Expectation: model_preparation event is captured in trace_view.json.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        prof = init_profiler(tmpdir)
        prof.start()
        net = create_resnet50()
        save_file = os.path.join(tmpdir, "resnet50.safetensors")
        ms.save_checkpoint(net, save_file, format="safetensors")
        dataset = create_dataset("/home/workspace/mindspore_dataset/cifar-10-batches-bin")
        optim = Momentum(net.get_parameters(), 0.01, 0.9)
        loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        model = Model(net, loss_fn=loss_fn, optimizer=optim)
        ms.load_checkpoint(save_file, net, format="safetensors")
        stop_cb = StopCallback(6)
        model.train(1, dataset, callbacks=stop_cb)
        prof.stop()
        trace_view_path = glob.glob(os.path.join(tmpdir, "*", "ASCEND_PROFILER_OUTPUT", "trace_view.json"))[0]
        with open(trace_view_path, 'r', encoding='utf-8') as f:
            trace_data = json.load(f)
        trace_content = json.dumps(trace_data)
        assert_list = ["model_preparation", "save_checkpoint", "load_checkpoint", "type_inference", "task_emit"]
        for item in assert_list:
            assert item in trace_content, f"{item} not found in trace_view.json"


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_mstx_range_decorator():
    """
    Feature: Test _mstx_range_decorator functionality
    Description: Test that the decorator correctly wraps functions
    Expectation: Decorated function executes normally and preserves metadata
    """
    def test_func(x, y=10):
        """Test function."""
        return x + y

    decorator = _mstx_range_decorator("test_message", domain="test_domain")
    assert callable(decorator), "Decorator factory should be callable"

    decorated_func = decorator(test_func)
    assert callable(decorated_func), "Decorated function should be callable"

    assert decorated_func.__name__ == "test_func", "Function name should be preserved"
    assert decorated_func.__doc__ == "Test function.", "Function docstring should be preserved"

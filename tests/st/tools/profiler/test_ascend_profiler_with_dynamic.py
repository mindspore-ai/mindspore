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
"""test ascend profiler with dynamic."""
import os
import glob
import json
import tempfile
import numpy as np
from tests.mark_utils import arg_mark
from model_zoo import TinyTransformer
from fake_dataset import FakeDataset
from file_check import FileChecker
import mindspore as ms
from mindspore import context, nn
import mindspore.dataset as ds
from mindspore.train import Model
from mindspore.profiler import DynamicProfilerMonitor
from mindspore.profiler.analysis.parser.base_parser import BaseParser
from mindspore.profiler.common.file_manager import FileManager


class StepMonitor(ms.Callback):
    def on_train_step_begin(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        print(f"-------------- Step {step_num} begin ----------------")

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        print(f"-------------- Step {step_num} end ----------------")


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.fc = nn.Dense(2, 2)

    def construct(self, x):
        return self.fc(x)


def generator_net():
    for _ in range(2):
        yield np.ones([2, 2]).astype(np.float32), np.ones([2]).astype(np.int32)


def train(net):
    optimizer = nn.Momentum(net.trainable_params(), 1, 0.9)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    data = ds.GeneratorDataset(generator_net(), ["data", "label"])
    model = ms.train.Model(net, loss, optimizer)
    model.train(1, data)


def train_net_with_dynamic_profiler_step(output_path, cfg_path):
    net = Net()
    STEP_NUM = 15
    context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
    dp = DynamicProfilerMonitor(cfg_path=cfg_path, output_path=output_path)
    for i in range(STEP_NUM):
        train(net)
        if i == 5:
            change_cfg_json(os.path.join(cfg_path, "profiler_config.json"))
        dp.step()


def change_cfg_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    data['start_step'] = 6
    data['stop_step'] = 7
    data['profiler_level'] = "Level1"
    data['aic_metrics'] = "MemoryUB"
    data['activities'] = ["CPU", "NPU"]

    FileManager.create_json_file(json_path, data)


def train_tiny_transformer_with_dynamic_profiler(output_path, cfg_path):
    ds_train = FakeDataset.create_fake_nlp_dataset(
        seq_len=1,
        batch_size=1,
        d_model=2,
        tgt_len=1,
        num_samples=5,
        num_parallel_workers=1
    )

    network = TinyTransformer(
        d_model=2,
        nhead=1,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=2
    )

    profile_callback = DynamicProfilerMonitor(cfg_path=cfg_path, output_path=output_path)
    step_cb = StepMonitor()
    model = Model(network)
    model.train(1, ds_train, callbacks=[profile_callback, step_cb], dataset_sink_mode=False)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.0)
def test_tiny_transformer_pynative_with_dynamic_profiler():
    """
    Feature: DynamicProfilerMonitor
    Description: Test the Ascend profiler in pynative mode with DynamicProfilerMonitor, using a static shape
                 model tiny transformer.
    Expectation: The profiler collects and analyzes data successfully, and the output files are correctly generated
                 in the temporary directory.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    data_cfg = {
        "start_step": 2,
        "stop_step": 3,
        "aic_metrics": 1,
        "profiler_level": -1,
        "profile_framework": 1,
        "analyse_mode": 0,
        "with_stack": True,
        "parallel_strategy": True,
        "data_simplification": False,
        "profile_memory": True,
        "analyse": True
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = os.path.join(tmpdir, "profiler_config.json")
        # set cfg file
        FileManager.create_json_file(cfg_path, data_cfg)

        rank_id = int(os.getenv('RANK_ID')) if os.getenv('RANK_ID') else 0
        train_tiny_transformer_with_dynamic_profiler(output_path=tmpdir, cfg_path=tmpdir)
        profiler_path = os.path.join(tmpdir, f"rank{rank_id}_start2_stop3")

        # Check trace_view.json
        trace_view_path = glob.glob(f"{profiler_path}/*_ascend_ms/"
                                    f"ASCEND_PROFILER_OUTPUT/trace_view.json")[0]
        FileChecker.check_timeline_values(
            trace_view_path,
            "name",
            [
                "mindspore/nn/*",  # check stack trace
                "PynativeFramework::*",  # check host trace
                "AscendCL@*",  # check CANN trace
                "aclnn*",  # check kernel on Ascend Hardware
                "Free",  # check overlay analysis
                "HostToDevice*",  # check HostToDevice flow
                "mindspore_to_npu"  # check mindspore_to_npu
            ],
            fuzzy_match=True
        )

        # check kernel_details.csv
        kernel_details_path = glob.glob(f"{profiler_path}/*_ascend_ms/"
                                        f"ASCEND_PROFILER_OUTPUT/kernel_details.csv")[0]
        FileChecker.check_csv_items(kernel_details_path, {"Name": ["*MatMul*", "LayerNorm*"]})

        # check step_trace_time.csv
        step_trace_time_path = glob.glob(f"{profiler_path}/*_ascend_ms/"
                                         f"ASCEND_PROFILER_OUTPUT/step_trace_time.csv")[0]
        FileChecker.check_file_line_count(step_trace_time_path, 3)
        # Check operate_memory.csv
        operate_memory_path = glob.glob(f"{profiler_path}/*_ascend_ms/"
                                        f"ASCEND_PROFILER_OUTPUT/operator_memory.csv")[0]
        FileChecker.check_csv_items(operate_memory_path, {"Name": ["*MatMul*"]})
        # Check profiler.log
        profiler_log_paths = glob.glob(f"{tmpdir}/*_ascend_ms/"
                                       f"logs/profiler_*.log")
        for profiler_log_path in profiler_log_paths:
            FileChecker.check_file_for_keyword(profiler_log_path, "error")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.00)
def test_tiny_transformer_kbk_with_dynamic_profiler():
    """
    Feature: DynamicProfilerMonitor
    Description: Test the Ascend profiler in KBK mode with DynamicProfilerMonitor, using a static shape
                 model tiny transformer.
    Expectation: The profiler collects and analyzes data successfully, and the output files are correctly generated
                 in the temporary directory.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_level="O0")
    BaseParser.EXEC_HOOK_TIMEOUT = 3 * 60
    data_cfg = {
        "start_step": 2,
        "stop_step": 3,
        "aic_metrics": 1,
        "profiler_level": -1,
        "profile_framework": 1,
        "analyse_mode": 0,
        "with_stack": True,
        "parallel_strategy": True,
        "data_simplification": False,
        "profile_memory": True,
        "analyse": True
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = os.path.join(tmpdir, "profiler_config.json")
        # set cfg file
        FileManager.create_json_file(cfg_path, data_cfg)

        rank_id = int(os.getenv('RANK_ID')) if os.getenv('RANK_ID') else 0
        train_tiny_transformer_with_dynamic_profiler(output_path=tmpdir, cfg_path=tmpdir)
        profiler_path = os.path.join(tmpdir, f"rank{rank_id}_start2_stop3")

        # Check trace_view.json
        trace_view_path = glob.glob(f"{profiler_path}/*_ascend_ms/"
                                    f"ASCEND_PROFILER_OUTPUT/trace_view.json")[0]
        FileChecker.check_timeline_values(
            trace_view_path,
            "name",
            [
                "mindspore/nn/*",  # check stack trace
                "Kernel::*",  # check host trace
                "AscendCL@*",  # check CANN trace
                "model-Transformer",  # check scope layer
                "aclnn*",  # check kernel on Ascend Hardware
                "Free",  # check overlay analysis
                "HostToDevice*",  # check HostToDevice flow
                "mindspore_to_npu"  # check mindspore_to_npu
            ],
            fuzzy_match=True
        )

        # check kernel_details.csv
        kernel_details_path = glob.glob(f"{profiler_path}/*_ascend_ms/"
                                        f"ASCEND_PROFILER_OUTPUT/kernel_details.csv")[0]
        FileChecker.check_csv_items(kernel_details_path, {"Name": ["*MatMul*", "LayerNorm*"]})

        # check step_trace_time.csv
        step_trace_time_path = glob.glob(f"{profiler_path}/*_ascend_ms/"
                                         f"ASCEND_PROFILER_OUTPUT/step_trace_time.csv")[0]
        FileChecker.check_file_line_count(step_trace_time_path, 3)
        # Check operate_memory.csv
        operate_memory_path = glob.glob(f"{profiler_path}/*_ascend_ms/"
                                        f"ASCEND_PROFILER_OUTPUT/operator_memory.csv")[0]
        FileChecker.check_csv_items(operate_memory_path, {"Name": ["Default"]}, fuzzy_match=True)
        # Check profiler.log
        profiler_log_paths = glob.glob(f"{tmpdir}/*_ascend_ms/"
                                       f"logs/profiler_*.log")
        for profiler_log_path in profiler_log_paths:
            FileChecker.check_file_for_keyword(profiler_log_path, "error")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1024.03)
def test_tiny_transformer_o2_with_dynamic_profiler():
    """
    Feature: DynamicProfilerMonitor
    Description: Test the Ascend profiler in GRAPH mode with DynamicProfilerMonitor, using a static shape
                 model tiny transformer.
    Expectation: The profiler collects and analyzes data successfully, and the output files are correctly generated
                 in the temporary directory.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_level="O2")
    BaseParser.EXEC_HOOK_TIMEOUT = 3 * 60
    data_cfg = {
        "start_step": 2,
        "stop_step": 3,
        "aic_metrics": 1,
        "profiler_level": -1,
        "profile_framework": 1,
        "analyse_mode": 0,
        "with_stack": True,
        "parallel_strategy": True,
        "data_simplification": False,
        "profile_memory": True,
        "analyse": True
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = os.path.join(tmpdir, "profiler_config.json")
        # set cfg file
        FileManager.create_json_file(cfg_path, data_cfg)

        rank_id = int(os.getenv('RANK_ID')) if os.getenv('RANK_ID') else 0
        train_tiny_transformer_with_dynamic_profiler(output_path=tmpdir, cfg_path=tmpdir)
        profiler_path = os.path.join(tmpdir, f"rank{rank_id}_start2_stop3")

        # Check trace_view.json
        trace_view_path = glob.glob(f"{profiler_path}/*_ascend_ms/"
                                    f"ASCEND_PROFILER_OUTPUT/trace_view.json")[0]
        FileChecker.check_timeline_values(
            trace_view_path,
            "name",
            [
                "mindspore/nn/*",  # check stack trace
                "Kernel::*",  # check host trace
                # "Model@ModelLoad",  # check CANN trace, 910a not support
                "model-Transformer",  # check scope layer
                "*MatMul*",  # check kernel on Ascend Hardware
                "Free",  # check overlay analysis
                "HostToDevice*",  # check HostToDevice flow
            ],
            fuzzy_match=True
        )

        # check kernel_details.csv
        kernel_details_path = glob.glob(f"{profiler_path}/*_ascend_ms/"
                                        f"ASCEND_PROFILER_OUTPUT/kernel_details.csv")[0]
        FileChecker.check_csv_items(kernel_details_path, {"Name": ["*MatMul*", "LayerNorm*"]})

        # check step_trace_time.csv
        step_trace_time_path = glob.glob(f"{profiler_path}/*_ascend_ms/"
                                         f"ASCEND_PROFILER_OUTPUT/step_trace_time.csv")[0]
        FileChecker.check_file_line_count(step_trace_time_path, 3)
        # Check operate_memory.csv
        operate_memory_path = glob.glob(f"{profiler_path}/*_ascend_ms/"
                                        f"ASCEND_PROFILER_OUTPUT/operator_memory.csv")[0]
        FileChecker.check_csv_items(operate_memory_path, {"Name": ["*Default*"]}, fuzzy_match=True)
        # Check profiler.log
        profiler_log_paths = glob.glob(f"{tmpdir}/*_ascend_ms/"
                                       f"logs/profiler_*.log")
        for profiler_log_path in profiler_log_paths:
            FileChecker.check_file_for_keyword(profiler_log_path, "error")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
def test_net_with_dynamic_profiler_step():
    """
    Feature: Dynamic Step Profiler Testing
    Description: This test function is designed to verify the functionality of the dynamic step profiler.
    Expectation: The test expects to find specific profiling data in the output files of the profiler.
    """
    data_cfg = {
        "start_step": 2,
        "stop_step": 5,
        "aic_metrics": 1,
        "profiler_level": -1,
        "profile_framework": 1,
        "analyse_mode": 0,
        "with_stack": True,
        "profile_memory": True,
        "parallel_strategy": True,
        "data_simplification": False,
        "analyse": True
    }
    with tempfile.TemporaryDirectory(suffix="_step_profiler") as tmpdir:
        cfg_path = os.path.join(tmpdir, "profiler_config.json")
        # set cfg file
        FileManager.create_json_file(cfg_path, data_cfg)

        rank_id = int(os.getenv('RANK_ID')) if os.getenv('RANK_ID') else 0
        train_net_with_dynamic_profiler_step(output_path=tmpdir, cfg_path=tmpdir)
        profiler_step_2_5_path = os.path.join(tmpdir, f"rank{rank_id}_start2_stop5")
        profiler_step_6_7_path = os.path.join(tmpdir, f"rank{rank_id}_start6_stop7")

        # Check trace_view.json
        trace_view_step_2_5_path = glob.glob(f"{profiler_step_2_5_path}/*_ascend_ms/"
                                             f"ASCEND_PROFILER_OUTPUT/trace_view.json")[0]
        trace_view_step_6_7_path = glob.glob(f"{profiler_step_6_7_path}/*_ascend_ms/"
                                             f"ASCEND_PROFILER_OUTPUT/trace_view.json")[0]
        FileChecker.check_timeline_values(
            trace_view_step_2_5_path,
            "name",
            ["*ProfilerStep#1",
             "*ProfilerStep#2",
             "*ProfilerStep#3",
             "*ProfilerStep#4",
             "*MatMul*",
             "*Add*"
             ],
            fuzzy_match=True
        )
        FileChecker.check_timeline_values(
            trace_view_step_6_7_path,
            "name",
            ["*ProfilerStep#1",
             "*ProfilerStep#2",
             "*MatMul*",
             "*Add*"],
            fuzzy_match=True
        )
        # Check kernel_details.csv
        kernel_details_step_2_5_path = glob.glob(f"{profiler_step_2_5_path}/*_ascend_ms/"
                                                 f"ASCEND_PROFILER_OUTPUT/kernel_details.csv")[0]
        kernel_details_step_6_7_path = glob.glob(f"{profiler_step_6_7_path}/*_ascend_ms/"
                                                 f"ASCEND_PROFILER_OUTPUT/kernel_details.csv")[0]
        FileChecker.check_csv_items(kernel_details_step_2_5_path, {"Step ID": ["1", "2", "3", "4"]},
                                    fuzzy_match=False
                                    )
        FileChecker.check_csv_items(kernel_details_step_2_5_path, {"Name": ["*BiasAdd*", "*MatMul*"]})
        FileChecker.check_csv_items(kernel_details_step_6_7_path, {"Step ID": ["1", "2"]},
                                    fuzzy_match=False
                                    )
        FileChecker.check_csv_items(kernel_details_step_6_7_path, {"Name": ["*BiasAdd*", "*MatMul"]})
        FileChecker.check_csv_headers(kernel_details_step_6_7_path, ["aic_ub_read_bw_scalar(GB/s)",
                                                                     "aic_ub_write_bw_scalar(GB/s)",
                                                                     "aiv_ub_read_bw_vector(GB/s)",
                                                                     "aiv_ub_write_bw_vector(GB/s)",
                                                                     "aiv_ub_read_bw_scalar(GB/s)",
                                                                     "aiv_ub_write_bw_scalar(GB/s)"])
        # Check operate_memory.csv
        operate_memory_2_5_path = glob.glob(f"{profiler_step_2_5_path}/*_ascend_ms/"
                                            f"ASCEND_PROFILER_OUTPUT/operator_memory.csv")[0]
        operate_memory_6_7_path = glob.glob(f"{profiler_step_6_7_path}/*_ascend_ms/"
                                            f"ASCEND_PROFILER_OUTPUT/operator_memory.csv")[0]
        FileChecker.check_csv_items(operate_memory_2_5_path, {"Name": ["*Default*"]}, fuzzy_match=True)
        FileChecker.check_csv_items(operate_memory_6_7_path, {"Name": ["*Default*"]}, fuzzy_match=True)
        # Check profiler.log
        profiler_log_paths = glob.glob(f"{tmpdir}/*_ascend_ms/"
                                       f"logs/profiler_*.log")
        for profiler_log_path in profiler_log_paths:
            FileChecker.check_file_for_keyword(profiler_log_path, "error")

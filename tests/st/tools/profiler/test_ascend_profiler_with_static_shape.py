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
import pytest
import os
import glob
import tempfile
import mindspore
from mindspore import context
from mindspore import Profiler
from mindspore.profiler import ProfilerLevel, ExportType
from mindspore.profiler.analysis.parser.base_parser import BaseParser

from tests.mark_utils import arg_mark
from file_check import FileChecker
from model_zoo import TinyTransformer
from fake_dataset import FakeDataset


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=1024.03)
def test_ascend_graph_mode_profiler_with_static_shape_all_parameters_on():
    """
    Feature: Ascend Graph Mode Profiler with All Parameters Enabled
    Description: Test the Ascend profiler in graph mode with all profiling parameters turned on, using a static shape
                 model.
    Expectation: The profiler collects and analyzes data successfully, and the output files are correctly generated
                 in the temporary directory.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O2"})
    BaseParser.EXEC_HOOK_TIMEOUT = 3 * 60
    with tempfile.TemporaryDirectory() as tmpdir:
        rank_id = int(os.getenv('RANK_ID')) if os.getenv('RANK_ID') else 0
        profiler = Profiler(
            profiler_level=ProfilerLevel.Level1,
            output_path=tmpdir,
            profile_memory=True,
            l2_cache=True,
            hbm_ddr=True,
            pcie=True,
            sys_io=True,
            sys_interconnection=True,
            sync_enable=True,
            data_process=True,
            data_simplification=False
        )
        net = TinyTransformer(d_model=2, nhead=1, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=4)
        nlp_dataset = FakeDataset.create_fake_nlp_dataset(seq_len=1, batch_size=1, d_model=2, tgt_len=1, num_samples=1)
        for src, tgt in nlp_dataset:
            net(src, tgt)
        profiler.analyse()
        check_ascend_profiler_graph_files(tmpdir, rank_id)


@pytest.mark.skip(reason="View feature no support")
@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_ascend_pynative_mode_profiler_with_static_shape_all_parameters_on():
    """
    Feature: Ascend pynative Mode Profiler with All Parameters Enabled
    Description: Test the Ascend profiler in pynative mode with all profiling parameters turned on, using a static shape
                 model.
    Expectation: The profiler collects and analyzes data successfully, and the output files are correctly generated
                 in the temporary directory.
    """
    # pylint: disable=protected-access
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    BaseParser.EXEC_HOOK_TIMEOUT = 3 * 60
    with tempfile.TemporaryDirectory() as tmpdir:
        rank_id = int(os.getenv('RANK_ID')) if os.getenv('RANK_ID') else 0
        experimental_config = mindspore.profiler._ExperimentalConfig(profiler_level=ProfilerLevel.Level1,
                                                                     l2_cache=True,
                                                                     sys_io=True,
                                                                     sys_interconnection=True,
                                                                     data_simplification=False,
                                                                     export_type=[ExportType.Text])
        with mindspore.profiler.profile(data_process=True,
                                        profile_memory=True,
                                        hbm_ddr=True,
                                        record_shapes=True,
                                        schedule=mindspore.profiler.schedule(wait=0, warmup=0,
                                                                             active=1, repeat=1, skip_first=0),
                                        on_trace_ready=mindspore.profiler.tensorboard_trace_handler(dir_name=tmpdir),
                                        experimental_config=experimental_config) as prof:
            net = TinyTransformer(d_model=2, nhead=1, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=4)
            nlp_dataset = FakeDataset.create_fake_nlp_dataset(seq_len=1, batch_size=1,
                                                              d_model=2, tgt_len=1, num_samples=1)
            for src, tgt in nlp_dataset:
                net(src, tgt)
                prof.step()
        check_ascend_profiler_pynative_files(tmpdir, rank_id)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=1.00)
def test_ascend_kbk_mode_profiler_with_static_shape_all_parameters_on():
    """
    Feature: Ascend kbk Mode Profiler with All Parameters Enabled
    Description: Test the Ascend profiler in kbk mode with all profiling parameters turned on, using a static shape
                 model.
    Expectation: The profiler collects and analyzes data successfully, and the output files are correctly generated
                 in the temporary directory.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0"})
    BaseParser.EXEC_HOOK_TIMEOUT = 3 * 60
    # pylint: disable=protected-access
    with tempfile.TemporaryDirectory() as tmpdir:
        rank_id = int(os.getenv('RANK_ID')) if os.getenv('RANK_ID') else 0
        experimental_config = mindspore.profiler._ExperimentalConfig(profiler_level=ProfilerLevel.Level1,
                                                                     l2_cache=True,
                                                                     sys_io=True,
                                                                     sys_interconnection=True,
                                                                     data_simplification=False,
                                                                     export_type=[ExportType.Text])
        with mindspore.profiler.profile(data_process=True,
                                        profile_memory=True,
                                        hbm_ddr=True,
                                        record_shapes=True,
                                        schedule=mindspore.profiler.schedule(wait=0, warmup=0,
                                                                             active=1, repeat=1, skip_first=0),
                                        on_trace_ready=mindspore.profiler.tensorboard_trace_handler(dir_name=tmpdir),
                                        experimental_config=experimental_config) as prof:
            net = TinyTransformer(d_model=2, nhead=1, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=4)
            nlp_dataset = FakeDataset.create_fake_nlp_dataset(seq_len=1, batch_size=1,
                                                              d_model=2, tgt_len=1, num_samples=1)
            for src, tgt in nlp_dataset:
                net(src, tgt)
                prof.step()
        check_ascend_profiler_kbk_files(tmpdir, rank_id)


def check_ascend_profiler_all_parameters_on_common_files(profiler_path: str, rank_id: int):
    ascend_profiler_output_path = glob.glob(f"{profiler_path}/*_ascend_ms/ASCEND_PROFILER_OUTPUT")[0]
    ascend_ms_dir = glob.glob(f"{profiler_path}/*_ascend_ms")[0]
    msprof_db_path = glob.glob(f"{ascend_ms_dir}/PROF_*/*.db")

    # check db in PROF_* is deleted
    assert not msprof_db_path, f"Expect no db file in PROF_*, but got {len(msprof_db_path)}"

    # check hbm*.csv
    hbm_path = glob.glob(f"{ascend_profiler_output_path}/hbm*")[0]
    FileChecker.check_csv_headers(hbm_path, ["Device_id", "Metric", "Read(MB/s)", "Write(MB/s)"])

    # check l2cache*.csv
    l2_cache_path = glob.glob(f"{ascend_profiler_output_path}/l2_cache*")[0]
    FileChecker.check_csv_items(l2_cache_path, {
        "Op Name": ["*Add*", "*MatMul*", "*LayerNorm*"]
    })

    # check pcie*.csv
    pcie_path = glob.glob(f"{ascend_profiler_output_path}/pcie*")[0]
    FileChecker.check_csv_headers(pcie_path, ["Device_id", "Mode", "Min", "Max", "Avg"])

    # check hccs*.csv
    hccs_path = glob.glob(f"{ascend_profiler_output_path}/hccs*")[0]
    FileChecker.check_csv_headers(hccs_path, ["Device_id", "Mode", "Max", "Min", "Average"])

    # check nic*.csv
    nic_path = glob.glob(f"{ascend_profiler_output_path}/nic*")[0]
    FileChecker.check_csv_headers(nic_path, ["Device_id", "Bandwidth(MB/s)",
                                             "Rx Bandwidth efficiency(%)", "rxPacket/s"])

    # check roce*.csv
    roce_path = glob.glob(f"{ascend_profiler_output_path}/roce*")[0]
    FileChecker.check_csv_headers(roce_path, ["Device_id", "Bandwidth(MB/s)", "Rx Bandwidth efficiency(%)"])

    # Check trace_view.json
    trace_view_path = os.path.join(ascend_profiler_output_path, "trace_view.json")
    FileChecker.check_timeline_values(
        trace_view_path,
        "name",
        [
            "Dataset::*",            # check dataset trace
            "PynativeFramework::*",  # check host trace
            "AscendCL@*",            # check CANN trace
            "GetNext*",              # check kernel on Ascend Hardware
            "Free",                  # check overlay analysis
            "HostToDevice*",         # check HostToDevice flow
        ],
        fuzzy_match=True
    )

    # check op_statistic.csv
    op_statistic_path = os.path.join(ascend_profiler_output_path, "op_statistic.csv")
    FileChecker.check_csv_items(op_statistic_path, {
        "OP Type": ["*Add*", "*MatMul*", "*LayerNorm*"]
    })

    # check kernel_details.csv
    kernel_details_path = os.path.join(ascend_profiler_output_path, "kernel_details.csv")
    FileChecker.check_csv_items(kernel_details_path, {
        "Name": ["*Add*", "*MatMul*", "*LayerNorm*"]
    })

    # check profile_info_*.json
    profile_info_path = os.path.join(ascend_ms_dir, f"profiler_info_{rank_id}.json")
    FileChecker.check_json_items(profile_info_path, {
        "profiler_parameters.with_stack": False,
        "profiler_parameters.profile_memory": True,
        "profiler_parameters.data_simplification": False
    })

    # check dataset.csv
    dataset_path = os.path.join(ascend_profiler_output_path, f"dataset.csv")
    FileChecker.check_csv_items(dataset_path, {
        "Operation": ["Pipeline", "RandomDataOp"]
    })


def check_ascend_profiler_graph_files(profiler_path: str, rank_id: int):
    ascend_profiler_output_path = glob.glob(f"{profiler_path}/*_ascend_ms/ASCEND_PROFILER_OUTPUT")[0]

    # check operate_memory.csv
    operate_memory_path = os.path.join(ascend_profiler_output_path, "operator_memory.csv")
    FileChecker.check_csv_items(operate_memory_path, {
        "Name": ["model.encoder*", "model.decoder*"]
    })

    # check static_op_mem.csv
    static_op_mem_path = os.path.join(ascend_profiler_output_path, "static_op_mem.csv")
    FileChecker.check_csv_items(static_op_mem_path, {
        "Op Name": ["*Add*", "*LayerNorm*"]
    })


def check_ascend_profiler_pynative_files(profiler_path: str, rank_id: int):
    check_ascend_profiler_all_parameters_on_common_files(profiler_path, rank_id)
    ascend_profiler_output_path = glob.glob(f"{profiler_path}/*_ascend_ms/ASCEND_PROFILER_OUTPUT")[0]

    # check operate_memory.csv
    operate_memory_path = os.path.join(ascend_profiler_output_path, "operator_memory.csv")
    FileChecker.check_csv_items(operate_memory_path, {
        "Name": ["*Add*", "*Sqrt*", "*LayerNorm*"]
    })
    operator_details_path = os.path.join(ascend_profiler_output_path, "operator_details.csv")
    FileChecker.check_csv_items(operator_details_path, {
        "Name": ["Transpose", "MatMulExt", "AddExt"],
        "Input Shapes": ["6,2", "1,1,2;2,6", "1,1,6;6"]
    })


def check_ascend_profiler_kbk_files(profiler_path: str, rank_id: int):
    check_ascend_profiler_all_parameters_on_common_files(profiler_path, rank_id)
    ascend_profiler_output_path = glob.glob(f"{profiler_path}/*_ascend_ms/ASCEND_PROFILER_OUTPUT")[0]

    # check operate_memory.csv
    operate_memory_path = os.path.join(ascend_profiler_output_path, "operator_memory.csv")
    FileChecker.check_csv_items(operate_memory_path, {
        "Name": ["*Add*", "*MatMul*", "*LayerNorm*"]
    })
    operator_details_path = os.path.join(ascend_profiler_output_path, "operator_details.csv")
    FileChecker.check_csv_items(operator_details_path, {
        "Name": ["Split", "Transpose", "MatMulExt"],
        "Input Shapes": ["6,2;;", "2,2;2", "1,1,2;2,2"]
    })

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
"""profiler with base function"""
import os
import tempfile

from tests.st.tools.profiler.daily_test.profiler_check import MSProfilerChecker, get_directory_from_check_dir
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_func_profiler_cluster_pipeline_parallel_file_check_01():
    """
    Feature: Profiler cluster/pipeline-parallel file check
    Description: Run cluster pipeline-parallel and validate profiling directory layout and outputs.
    Expectation: Produce level-2 data including HBM/PCIE metrics and metadata.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        prof_dir = os.path.join(tmpdir, "prof_dir")
        ret = os.system(f"python run_resnet_with_cluster_pipeline_parallel.py --prof_dir={prof_dir} ")
        assert ret == 0
        directory = get_directory_from_check_dir(prof_dir)
        check_dir_num = len(directory)
        for i in range(check_dir_num):
            prof_config = {
                "profiler_memory": True,
                "profile_level": 2,
                "hbm": True,
                "pcie": True,
                "add_metadata": True,
                "output_path": os.path.join(prof_dir, directory[i]),
            }
            profiler_check = MSProfilerChecker(prof_config, 1)
            profiler_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_func_profiler_with_schedule_on_trace_ready_use_default_repeat():
    """
    Feature: Profiler schedule on trace-ready (default repeat)
    Description: Trigger profiling on trace-ready using the default repeat count.
    Expectation: Generate level-1 outputs; verify steps 2 and 3.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        prof_dir = os.path.join(tmpdir, "prof_dir")
        ret = os.system(f"python run_resnet_with_schedule.py --prof_dir={prof_dir} ")
        assert ret == 0
        directory = get_directory_from_check_dir(prof_dir)
        check_dir_num = len(directory)
        for i in range(check_dir_num):
            prof_config = {
                "profiler_memory": True,
                "profile_level": 1,
                "pynative_step": True,
                "output_path": os.path.join(prof_dir, directory[i]),
            }
            profiler_check = MSProfilerChecker(prof_config, 1, check_step_id=[2, ])
            profiler_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_func_profiler_exper_cfg_set_level1_export_type_teext_db_default_schedule():
    """
    Feature: Profiler level-1 config, export text/DB, default schedule
    Description: Set level 1 with dbtext export and enable AICore L2Cache metrics.
    Expectation: Level-1 results produced with text/DB exports; step0 passes validation.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        prof_dir = os.path.join(tmpdir, "prof_dir")
        ret = os.system(f"python run_resnet_with_exper.py --prof_dir={prof_dir} ")
        assert ret == 0
        directory = get_directory_from_check_dir(prof_dir)
        check_dir_num = len(directory)
        for i in range(check_dir_num):
            prof_config = {
                "profiler_memory": True,
                "profile_level": 1,
                "pynative_step": True,
                "export_type": "dbtext",
                "aicore_metrics": "L2Cache",
                "output_path": os.path.join(prof_dir, directory[i]),
            }
            profiler_check = MSProfilerChecker(prof_config, 1, check_step_id=[0], repeat=1,
                                               step_numbers=["step0"], active=1)
            profiler_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_func_profiler_model_train_callback_exper_cfg_set_level0_add_metadata():
    """
    Feature: Profiler level-0 config with model-train callback metadata
    Description: Enable level 0 and add metadata with abnormal stack flags via training callback.
    Expectation: Level-0 profiling includes metadata and passes step0 validation.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        prof_dir = os.path.join(tmpdir, "prof_dir")
        ret = os.system(f"python run_resnet_with_model_train.py --prof_dir={prof_dir} ")
        assert ret == 0
        directory = get_directory_from_check_dir(prof_dir)
        check_dir_num = len(directory)
        for i in range(check_dir_num):
            prof_config = {
                "profiler_memory": True,
                "profile_level": 0,
                "add_metadata": True,
                "with_stack_not_normal": True,
                "pynative_step": True,
                "output_path": os.path.join(prof_dir, directory[i]),
            }
            profiler_check = MSProfilerChecker(prof_config, 1, check_step_id=[0], repeat=1,
                                               step_numbers=["step0"], active=1)
            profiler_check()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_profiler_llama2_train_001():
    """
    Feature: Profiler resnet training (step-wise)
    Description: Single-card training with memory profiling; collect and validate step outputs.
    Expectation: Produce profiling data for steps 3–7 and pass validation.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        prof_dir = os.path.join(tmpdir, "prof_dir")
        ret = os.system(f"python run_resnet_with_step.py --prof_dir={prof_dir} ")
        assert ret == 0
        directory = get_directory_from_check_dir(prof_dir)
        check_dir_num = len(directory)
        for i in range(check_dir_num):
            prof_config = {
                "profiler_memory": True,
                "profile_level": 0,
                "add_metadata": True,
                "pynative_step": True,
                "output_path": os.path.join(prof_dir, directory[i]),
            }
            profiler_check = MSProfilerChecker(prof_config, 1, check_step_id=[3, 4, 5, 6, 7])
            profiler_check()

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
"""profile env"""

import os
import tempfile
import json

from tests.mark_utils import arg_mark
from tests.st.tools.profiler.daily_test.profiler_check import MSProfilerChecker


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_func_profiler_env_normal():
    """
    Feature: Profiler test
    Description: set profile env normal
    Expectation: run success.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        original_env = os.environ.get("MS_PROFILER_OPTIONS")
        try:
            profiler_dir = os.path.join(tmpdir, "data")
            profiler_options = {
                "start": True,
                "output_path": profiler_dir,
                "profile_memory": True,
                "aicore_metrics": "AicoreNone",
                "l2_cache": False,
                "data_process": True,
                "profiler_level": "Level1",
                "activities": ["CPU", "NPU"]
            }
            profiler_options_str = json.dumps(profiler_options)

            ret = os.system(
                f"export MS_PROFILER_OPTIONS='{profiler_options_str}'; "
                f"python ./run_net.py --target=Ascend --mode=0"
            )
            assert ret == 0
            prof_config = {"output_path": profiler_dir,
                           "profile_memory": True,
                           "profile_level": 1}
            profiler_check = MSProfilerChecker(prof_config, 1)
            profiler_check()
        finally:
            if original_env is None:
                os.environ.pop("MS_PROFILER_OPTIONS", None)
            else:
                os.environ["MS_PROFILER_OPTIONS"] = original_env


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_func_profiler_env_parameters_start_normal():
    """
    Feature: Profiler test
    Description: set profile env normal
    Expectation: run success.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        original_env = os.environ.get("MS_PROFILER_OPTIONS")
        try:
            profiler_dir = os.path.join(tmpdir, "data")
            profiler_options = {
                "start": True,
                "output_path": profiler_dir,
                "profile_memory": True,
                "aicore_metrics": "AicoreNone",
                "l2_cache": True,
                "data_process": True,
                "profiler_level": "Level2",
                "with_stack": True,
                "activities": ["CPU", "NPU"],
                "export_type": ["text", "db"]
            }
            profiler_options_str = json.dumps(profiler_options)

            ret = os.system(
                f"export MS_PROFILER_OPTIONS='{profiler_options_str}'; "
                f"python ./run_net.py --target=Ascend --mode=0"
            )
            assert ret == 0
            prof_config = {"output_path": profiler_dir,
                           "profile_memory": True,
                           "profile_level": 2,
                           "export_type": "dbtext"}
            profiler_check = MSProfilerChecker(prof_config, 1)
            profiler_check()
        finally:
            if original_env is None:
                os.environ.pop("MS_PROFILER_OPTIONS", None)
            else:
                os.environ["MS_PROFILER_OPTIONS"] = original_env


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_func_profiler_env_parameters_op_time_false():
    """
    Feature: Profiler test
    Description: set profile env normal
    Expectation: run success.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        original_env = os.environ.get("MS_PROFILER_OPTIONS")
        try:
            profiler_dir = os.path.join(tmpdir, "data")
            profiler_options = {
                "start": True,
                "output_path": profiler_dir,
                "profile_memory": True,
                "aicore_metrics": "ArithmeticUtilization",
                "l2_cache": False,
                "data_process": True
            }
            ret = os.system(
                f"export MS_PROFILER_OPTIONS='{json.dumps(profiler_options)}'; "
                f"python ./run_net.py --target=Ascend --mode=0"
            )
            assert ret == 0
            prof_config = {"output_path": profiler_dir,
                           "profile_memory": True,
                           "profile_level": 0}
            profiler_check = MSProfilerChecker(prof_config, 1)
            profiler_check()
        finally:
            if original_env is None:
                os.environ.pop("MS_PROFILER_OPTIONS", None)
            else:
                os.environ["MS_PROFILER_OPTIONS"] = original_env

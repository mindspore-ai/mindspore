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

""" define marks """
import inspect
import pytest
import os
import shutil
import subprocess
from tests.st.common.random_generator import set_numpy_global_seed


def arg_mark(plat_marks, level_mark, card_mark, essential_mark, mem_peak=None):
    """
    Decorator for marking test cases with various attributes.
    Args:
        plat_marks (list): Platform marks like 'platform_ascend', 'platform_gpu'
        level_mark (str): Test level mark like 'level0', 'level1'
        card_mark (str): Card mark like 'onecard', 'allcards'
        essential_mark (str): Essential mark like 'essential', 'unessential'
        mem_peak (float, optional): Expected peak memory usage in MB
    Returns:
        function: Decorator function
    """
    optional_plat_marks = ['platform_ascend', 'platform_ascend910b', 'platform_ascend310p', 'platform_gpu',
                           'cpu_linux', 'cpu_windows', 'cpu_macos']
    optional_level_marks = ['level0', 'level1', 'level2', 'level3', 'level4']
    optional_card_marks = ['onecard', 'allcards', 'dryrun', 'dryrun_only']
    optional_essential_marks = ['essential', 'unessential']
    if not plat_marks or not set(plat_marks).issubset(set(optional_plat_marks)):
        raise ValueError("wrong plat_marks values")
    if level_mark not in optional_level_marks:
        raise ValueError("wrong level_mark value")
    if card_mark not in optional_card_marks:
        raise ValueError("wrong card_mark value")
    if essential_mark not in optional_essential_marks:
        raise ValueError("wrong essential_mark value")
    if card_mark == "onecard" and "ascend" in str(plat_marks) and level_mark in ['level0', 'level1'] and not mem_peak:
        raise ValueError("wrong mem_peak value")

    def decorator(func):
        for plat_mark in plat_marks:
            func = getattr(pytest.mark, plat_mark)(func)
        func = getattr(pytest.mark, level_mark)(func)
        func = getattr(pytest.mark, card_mark)(func)
        func = getattr(pytest.mark, essential_mark)(func)
        if mem_peak: 
            func = pytest.mark.mem_peak(mem_peak)(func)
        set_numpy_global_seed()
        file_path = inspect.getfile(func)
        if card_mark == "onecard" and "ascend" in str(plat_marks) and level_mark in ['level0', 'level1'] and \
                shutil.which("npu-smi"):
            case_path = "tests/" + file_path.split("tests/")[-1]
            case_root_path = file_path.split("/tests/")[0]
            qual_name = (func.__qualname__).replace('.', '::')
            cmd = f"echo {case_path}::{qual_name},{plat_marks},{level_mark},{card_mark},{mem_peak} >>\
                    {case_root_path}/tests_mark_info.txt"
            subprocess.run(cmd, timeout=5, shell=True)
        return func

    return decorator

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
"""Test with StreamCtx and event"""
import os
from tests.mark_utils import arg_mark


def clean_core_files(root_dir="."):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.startswith("core") and not filename.endswith((".py", ".txt", ".bin")):
                core_path = os.path.join(dirpath, filename)
                try:
                    os.remove(core_path)
                    print(f"Deleted core file: {core_path}")
                except OSError:
                    pass


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=6.00)
def test_single_withstream_multi_event():
    """
    Feature: Support event and with stream in graph mode.
    Description: Support event and with stream in graph mode. Check IR and mem_tracker.
    Expectation: Run success.
    """
    env = "export MS_ALLOC_CONF='memory_tracker:True'"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/run_with_stream_event.py"
    assert os.path.exists(script)
    case_name = "test_single_withstream_multi_event"
    output = real_path + "/test_single_withstream_multi_event.log"

    cmd = f"{env}; pytest -sv {script}::{case_name} > {output} 2>&1"
    return_code = os.system(cmd)

    assert os.path.exists(output)
    with open(output, "r", encoding='utf-8') as f:
        output_log = f.read()
        print(output_log, flush=True)
    assert return_code == 0
    assert "RaceChecker: Read error" not in output_log
    assert "RaceChecker: Write error" not in output_log


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=10.00)
def test_multi_withstream_single_event():
    """
    Feature: Support event and with stream in graph mode.
    Description: Support event and with stream in graph mode. Check IR and mem_tracker.
    Expectation: Run success.
    """
    env = "export MS_ALLOC_CONF='memory_tracker:True'"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/run_with_stream_event.py"
    assert os.path.exists(script)
    case_name = "test_multi_withstream_single_event"
    output = real_path + "/test_multi_withstream_single_event.log"

    cmd = f"{env}; pytest -sv {script}::{case_name} > {output} 2>&1"
    return_code = os.system(cmd)

    assert os.path.exists(output)
    with open(output, "r", encoding='utf-8') as f:
        output_log = f.read()
        print(output_log, flush=True)
    assert return_code == 0
    assert "RaceChecker: Read error" not in output_log
    assert "RaceChecker: Write error" not in output_log


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=10.00)
def test_multi_withstream_multi_event():
    """
    Feature: Support event and with stream in graph mode.
    Description: Support event and with stream in graph mode. Check IR and mem_tracker.
    Expectation: Run success.
    """
    env = "export MS_ALLOC_CONF='memory_tracker:True'"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/run_with_stream_event.py"
    assert os.path.exists(script)
    case_name = "test_multi_withstream_multi_event"
    output = real_path + "/test_multi_withstream_multi_event.log"

    cmd = f"{env}; pytest -sv {script}::{case_name} > {output} 2>&1"
    return_code = os.system(cmd)

    assert os.path.exists(output)
    with open(output, "r", encoding='utf-8') as f:
        output_log = f.read()
        print(output_log, flush=True)
    assert return_code == 0
    assert "RaceChecker: Read error" not in output_log
    assert "RaceChecker: Write error" not in output_log


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_eventwait_before_eventrecord():
    """
    Feature: Support event and with stream in graph mode.
    Description: Support event and with stream in graph mode. Check IR and mem_tracker.
    Expectation: Run success.
    """
    env = "export MS_ALLOC_CONF='memory_tracker:True'"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/run_with_stream_event.py"
    assert os.path.exists(script)
    case_name = "test_eventwait_before_eventrecord"
    output = real_path + "/test_eventwait_before_eventrecord.log"

    cmd = f"{env}; pytest -sv {script}::{case_name} > {output} 2>&1"
    os.system(cmd)

    assert os.path.exists(output)
    with open(output, "r", encoding='utf-8') as f:
        output_log = f.read()
        print(output_log, flush=True)
    skip_streamrecv_str = ("No paired StreamSend with event_id: 0 found, the launch of this node: "
                           "Default/StreamRecv-op0 will be skipped")
    assert skip_streamrecv_str in output_log
    assert "RaceChecker: Event id 0 is not found." in output_log
    clean_core_files(real_path)

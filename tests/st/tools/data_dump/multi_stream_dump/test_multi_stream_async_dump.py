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
"""multi stream sync and async dump statistic test case."""
import tempfile
import json
import concurrent.futures
import subprocess
from typing import List, Dict, Union
from tests.mark_utils import arg_mark
from tests.security_utils import security_off_wrap
from cmp_dump_statistic import compare_csv_files
from pathlib import Path

def run_script_with_args(script_path: str, args: List[str]) -> Dict[str, Union[str, int]]:
    """
    Non-blocking execution of a Shell script with command-line arguments
    :param script_path: Path to the Shell script
    :param args: List of command-line arguments for the script
    :return: Dictionary of execution results
    """
    cmd = ["bash", script_path] + args
    subprocess.run(
        cmd,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=240,
        check=True
    )


def generate_dump_json(dump_path, json_file_name, enable_sync):
    current_dir = Path(__file__).parent.parent
    json_path = current_dir / "test_e2e_statistic_config.json"
    with open(json_path, 'r', encoding="utf-8") as file:
        data = json.load(file)
        data["common_dump_settings"]["path"] = dump_path
        data["e2e_dump_settings"]["enable"] = enable_sync
        data["e2e_dump_settings"]["stat_calc_mode"] = "device"
    with open(json_file_name, 'w', encoding="utf-8") as f:
        json.dump(data, f)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='essential')
@security_off_wrap
def test_multi_stream_async_statistic_dump():
    """
    Feature: Multi-stream async dump statistic
    Description: Test the multi-stream async dump statistic functionality.
    Expectation: The test should pass without any errors.
    Steps:
        1. Generate JSON configuration files for sync and async dumps.
        2. Execute the dump script for sync and async dumps.
        3. Compare the results of sync and async dumps.
    """
    sh_path = str(Path(__file__).parent.absolute())
    with tempfile.TemporaryDirectory() as test_dir:
        base_path = Path(test_dir).absolute()
        # 1. Set paths and generate JSON configuration files
        enable_sync_list = [True, False]
        dump_path = [str(base_path / "sync_dump_data"), str(base_path / "async_dump_data")]
        dump_config_path = [str(base_path / "sync_dump_config.json"), str(base_path / "async_dump_config.json")]
        for dp, dc, enable_sync in zip(dump_path, dump_config_path, enable_sync_list):
            generate_dump_json(dump_path=dp, json_file_name=dc, enable_sync=enable_sync)
        scripts = [
            f"{sh_path}/msrun_sync.sh",
            f"{sh_path}/msrun_async.sh"
        ]
        scripts_args = [
            [dump_config_path[0]],
            [dump_config_path[1]]
        ]
        # 2. Execute the two Shell scripts concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(run_script_with_args, script, args)
                for script, args in zip(scripts, scripts_args)
            ]
        # 3. Wait for all concurrent tasks to complete and collect execution results
        concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)
        # 4. Result verification: Compare CSV files of sync and async data (validate data consistency)
        compare_csv_files(dump_path[0], dump_path[1])

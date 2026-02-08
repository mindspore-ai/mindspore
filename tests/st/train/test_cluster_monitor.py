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
import os
import shutil

from tests.mark_utils import arg_mark
from tests.st.export_and_load.test_remove_redundancy import set_port


def set_cluster_monitor_env(perf_dump_path, perf_dump_config):
    """Set environment variables required for cluster monitoring."""
    os.environ["PERF_DUMP_PATH"] = perf_dump_path
    os.environ["PERF_DUMP_CONFIG"] = perf_dump_config


def remove_cluster_monitor_env():
    """Remove cluster monitoring-related environment variables if they exist."""
    if "PERF_DUMP_PATH" in os.environ:
        del os.environ["PERF_DUMP_PATH"]
    if "PERF_DUMP_CONFIG" in os.environ:
        del os.environ["PERF_DUMP_CONFIG"]


def _check_log_content(cluster_path):
    """Check log files under the specified directory for specific strings."""
    if not os.path.isdir(cluster_path):
        raise ValueError(f"{cluster_path} does not exist.")
    for root, _, files in os.walk(cluster_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.log'):
                with open(file_path, 'r', encoding='utf-8') as log_file:
                    for line in log_file:
                        if "dp:" in line or "tp:" in line or "PP:" in line:
                            raise ValueError(
                                f"When dtpGroup is set to False, pp, dp and tp should not appear in the logs.")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards', essential_mark='essential')
def test_cluster_monitor_dtpGroup():
    """
    Feature: The perf_dump_config of Cluster monitor.
    Description: The dtpGroup configuration of perf_dump_config is set to false.
    Expectation: Success.
    """
    try:
        for i in range(8):
            os.mkdir(f"device{i}_cluster_monitor_dtpgroup")
        cluster_path = os.path.join(os.getcwd(), "cluster_monitor_log")
        os.mkdir(cluster_path)
        set_port(64833)
        set_cluster_monitor_env(cluster_path, "enable:true,steptime:true,dtpGroup:false")
        ret = os.system("msrun --worker_num=8 --local_worker_num=8 --join=True " \
                        "pytest -s cluster_monitor_case.py::test_cluster_monitor_dtpgroup_env")
        assert ret == 0
        _check_log_content(cluster_path)
    finally:
        remove_cluster_monitor_env()
        if os.path.exists(cluster_path):
            shutil.rmtree(cluster_path)
        for i in range(8):
            device_dir = f"device{i}_cluster_monitor_dtpgroup"
            if os.path.exists(device_dir):
                shutil.rmtree(device_dir)

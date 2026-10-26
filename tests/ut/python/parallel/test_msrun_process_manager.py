# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Tests for the scheduler selection logic of msrun."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from mindspore.parallel.cluster.process_entity import _api


def _create_args(node_rank, worker_num=8, local_worker_num=4, sim_level=-1):
    """Create the minimum msrun arguments required by _ProcessManager."""
    return SimpleNamespace(
        master_addr="127.0.0.1",
        master_port=8118,
        worker_num=worker_num,
        local_worker_num=local_worker_num,
        node_rank=node_rank,
        log_dir="",
        join=False,
        worker_log_name="",
        tail_worker_log="-1",
        cluster_time_out=600,
        bind_numa=False,
        bind_core=False,
        rank_table_file="",
        sim_level=sim_level,
        sim_rank_id=-1,
        task_script="test.py",
        task_script_args=[],
    )


def _create_manager(monkeypatch, *, node_rank, role=None, is_local=True, worker_num=8, local_worker_num=4,
                    sim_level=-1):
    """Create a process manager without launching a real process."""
    if sim_level == -1:
        monkeypatch.delenv("MS_SIMULATION_LEVEL", raising=False)
    else:
        monkeypatch.setenv("MS_SIMULATION_LEVEL", str(sim_level))
    if role is None:
        monkeypatch.delenv("MS_ROLE", raising=False)
    else:
        monkeypatch.setenv("MS_ROLE", role)

    taskd_handler = Mock()
    taskd_handler.check_enable.return_value = False
    args = _create_args(node_rank, worker_num, local_worker_num, sim_level)
    with patch.object(_api, "_is_local_ip", return_value=is_local), \
            patch.object(_api, "_generate_bind_context", return_value=Mock()), \
            patch.object(_api, "RSCPluginHandle", return_value=taskd_handler):
        return _api._ProcessManager(args)


@pytest.mark.parametrize(
    "node_rank, role, is_local, expected",
    [
        pytest.param(0, None, True, True, id="local_rank_zero"),
        pytest.param(1, None, True, False, id="local_nonzero_rank"),
        pytest.param(0, "MS_WORKER", True, False, id="worker_role"),
        pytest.param(0, "MS_SCHED", True, True, id="scheduler_role"),
        pytest.param(1, "MS_SCHED", True, True, id="scheduler_role_nonzero_rank"),
        pytest.param(-1, None, True, False, id="multi_node_rank_not_set"),
    ],
)
def test_is_master_requires_rank_zero_and_non_worker_role(
        monkeypatch, node_rank, role, is_local, expected):
    """
    Feature: Validate the msrun Scheduler master-node selection conditions.
    Description: Check node rank and process role combinations before local IP validation.
    Expectation: Only a rank-0 node without MS_ROLE=MS_WORKER is selected as master.
    """
    manager = _create_manager(monkeypatch, node_rank=node_rank, role=role, is_local=is_local)
    assert manager.is_master is expected


def test_master_with_non_local_address_raises(monkeypatch):
    """
    Feature: Validate the master address on the selected master node.
    Description: Create a rank-0 non-worker process whose master address is not local.
    Expectation: Process manager initialization raises RuntimeError.
    """
    with pytest.raises(RuntimeError, match="not a local IP address"):
        _create_manager(monkeypatch, node_rank=0, is_local=False)


def test_msrun_single_node_without_node_rank_is_implicit_rank_zero(monkeypatch):
    """
    Feature: Preserve single-node msrun Scheduler startup compatibility.
    Description: Create a single-node process manager without explicitly setting node_rank.
    Expectation: node_rank is treated as rank 0 and the node is selected as master.
    """
    manager = _create_manager(monkeypatch, node_rank=-1, is_local=True, worker_num=8, local_worker_num=8)
    assert manager.node_rank == 0
    assert manager.is_master


@pytest.mark.parametrize(
    "role, sim_level, expected",
    [
        pytest.param(None, -1, True, id="normal_master"),
        pytest.param("MS_WORKER", -1, False, id="worker_role"),
        pytest.param(None, 0, False, id="simulation_mode"),
    ],
)
def test_run_scheduler_launch_decision(monkeypatch, role, sim_level, expected):
    """
    Feature: Validate the msrun Scheduler launch branch.
    Description: Execute the process manager run path for normal, worker-role, and simulation scenarios.
    Expectation: start_scheduler is called only for a normal eligible master node.
    """
    manager = _create_manager(monkeypatch, node_rank=0, role=role, is_local=True, sim_level=sim_level)
    with patch.object(manager, "start_scheduler") as start_scheduler, \
            patch.object(manager, "start_workers"):
        manager.run()
    assert start_scheduler.called is expected

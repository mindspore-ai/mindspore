# Copyright 2023 Huawei Technologies Co., Ltd
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
"""API for ms_run"""
import os
import multiprocessing
import sys
from ._utils import _generate_cmd, _generate_url, _is_local_ip, _send_scale_num, _get_status_and_params

class _Node:
    """
    Base class for dynamic networking nodes.

    """
    def __init__(self, worker_num, sched_host, sched_port, entrypoint):
        self.worker_num = worker_num
        self.sched_host = sched_host
        self.sched_port = sched_port
        self.entrypoint = entrypoint

    def run(self):
        """
        Runs the node by setting environment variables and executing the entrypoint command or script.

        """
        os.environ["MS_WORKER_NUM"] = str(self.worker_num)
        os.environ["MS_SCHED_HOST"] = self.sched_host
        os.environ["MS_SCHED_PORT"] = str(self.sched_port)

class _MetaServerNode(_Node):
    """
    Scheduler node for dynamic networking. Inherits from the Node class.

    """
    def run(self):
        """
        Runs the MetaServerNode by setting environment variables, setting the MS_ROLE variable to
         "MS_SCHED",  and executing the entrypoint command or script.

        """
        super().run()
        os.environ["MS_ROLE"] = "MS_SCHED"
        os.system(self.entrypoint)

class _ComputeGraphNode(_Node):
    """
    Worker node for dynamic networking. Inherits from the Node class.

    """
    def __init__(self, worker_num, sched_host, sched_port, node_id, entrypoint):
        super().__init__(worker_num, sched_host, sched_port, entrypoint)
        self.node_id = node_id


    def run(self):
        """
        Runs the ComputeGraphNode by setting environment variables, setting the MS_NODE_ID variable
        to the node ID, setting the MS_ROLE variable to "MS_WORKER", and executing the entrypoint
        command or script.

        """
        super().run()
        os.environ["MS_NODE_ID"] = str(self.node_id)
        os.environ["MS_ROLE"] = "MS_WORKER"
        os.system(self.entrypoint)


class _ProcessManager:
    """
    Manages the local dynamic networking process. Responsible for dynamic networking and elastic
    training

    """
    def __init__(self, args):
        """
        Initializes a ProcessManager object.

        Args:
            args: An object containing the command-line arguments.

        """
        self.processes = []

        """`is_master` flags whether the current node is the master node."""
        self.is_master = _is_local_ip(args.master_addr)

        self.master_addr = args.master_addr
        self.master_port = args.master_port

        self.total_nodes = args.total_nodes
        self.local_nodes = args.local_nodes

        self.cmd = args.training_script
        self.cmd_args = args.training_script_args

        """`is_scale` flags whether the current task is a scaling task and there is already a
        manager on the current node."""
        self.is_scale = False
        if args.is_scalein:
            self.is_scale = True
            self.scale_num = -args.scale_num
        elif args.is_scaleout:
            self.is_scale = True
            self.scale_num = args.scale_num

        self.scheduler_url = _generate_url(self.master_addr, self.master_port)

    def run(self):
        """
        Runs the process manager.

        """
        if self.is_scale:
            response_message = _send_scale_num(self.scheduler_url, self.scale_num)
            is_first_manager = response_message
            if is_first_manager:
                self.local_nodes = 0
            else:
                sys.exit()
        else:
            if self.is_master:
                self.start_scheduler()
        self.start_workers()

    def start_scheduler(self):
        """
        Starts the scheduler node.

        """
        msn = _MetaServerNode(self.total_nodes, self.master_addr, self.master_port,
                              _generate_cmd(self.cmd, self.cmd_args, "scheduler"))
        process = multiprocessing.Process(target=msn.run)
        self.processes.append(process)
        process.start()

    def start_workers(self):
        """
        Starts the worker nodes.

        """
        for i in range(self.local_nodes):
            cgn = _ComputeGraphNode(self.total_nodes, self.master_addr, self.master_port,
                                    i, _generate_cmd(self.cmd, self.cmd_args, "worker_"+str(i)))
            process = multiprocessing.Process(target=cgn.run)
            self.processes.append(process)
            process.start()

    def heartbeat_with_scheduler(self):
        """
        Sends a heartbeat to the scheduler and updates the total_nodes and local_nodes.

        Returns:
            bool: True if the network has changed, False otherwise.

        """
        network_changed, total_nodes, local_nodes = _get_status_and_params(self.scheduler_url)
        self.total_nodes = total_nodes
        self.local_nodes = local_nodes
        return network_changed

    def stop_processes(self):
        """
        Stops all running processes.

        """
        for process in self.processes:
            process.terminate()
            process.join()

    def stop_and_restart(self):
        """
        Stops all running processes and restarts the scheduler and workers.

        """
        self.stop_processes()
        if self.is_master:
            self.start_scheduler()
        self.start_workers()

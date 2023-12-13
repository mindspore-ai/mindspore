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
import re
import sys
import subprocess
import mindspore.log as logger
from ._utils import _generate_cmd_args_list, _generate_url, _is_local_ip, _send_scale_num,\
                    _get_status_and_params

class _Node:
    """
    Base class for dynamic networking nodes.

    """
    def __init__(self, worker_num, sched_host, sched_port, args_list, output_file):
        self.worker_num = worker_num
        self.sched_host = sched_host
        self.sched_port = sched_port
        self.args_list = args_list
        self.output_file = output_file

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
        with open(self.output_file, "w") as file_handle:
            return subprocess.Popen(self.args_list, stdout=file_handle, stderr=subprocess.STDOUT)

class _ComputeGraphNode(_Node):
    """
    Worker node for dynamic networking. Inherits from the Node class.
    """
    def __init__(self, worker_num, sched_host, sched_port, node_id, args_list, output_file):
        super().__init__(worker_num, sched_host, sched_port, args_list, output_file)
        self.node_id = node_id


    def run(self):
        """
        Runs the ComputeGraphNode by setting environment variables, setting the MS_NODE_ID variable
        to the node ID, setting the MS_ROLE variable to "MS_WORKER", and executing the entrypoint
        command or script.

        """
        super().run()
        if self.node_id is not None:
            os.environ["MS_NODE_ID"] = str(self.node_id)
        os.environ["MS_ROLE"] = "MS_WORKER"
        with open(self.output_file, "w") as file_handle:
            return subprocess.Popen(self.args_list, stdout=file_handle, stderr=subprocess.STDOUT)


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
        self.msn_process = []
        self.cgn_processes = []

        """`is_master` flags whether the current node is the master node."""
        self.is_master = _is_local_ip(args.master_addr)

        self.master_addr = args.master_addr
        self.master_port = args.master_port

        self.worker_num = args.worker_num
        self.local_worker_num = args.local_worker_num
        self.node_rank = args.node_rank

        self.log_dir = args.log_dir
        self.join = args.join

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

        # Create log directory and set the permission if not exists.
        if self.log_dir and not os.path.exists(self.log_dir):
            try:
                permissions = os.R_OK | os.W_OK | os.X_OK
                origin_mask = os.umask(permissions << 3 | permissions)
                mode = permissions << 6
                os.makedirs(self.log_dir, mode=mode, exist_ok=True)
            finally:
                os.umask(origin_mask)

    def run(self):
        """
        Runs the process manager.

        """
        if self.is_scale:
            response_message = _send_scale_num(self.scheduler_url, self.scale_num)
            is_first_manager = response_message
            if is_first_manager:
                self.local_worker_num = 0
            else:
                sys.exit()
        else:
            if self.is_master:
                self.start_scheduler()
        self.start_workers()

        if self.join:
            logger.warning("Distributed job is spawned. Waiting all processes to exit...")
            self.join_processes()

    def start_scheduler(self):
        """
        Starts the scheduler node.

        """
        msn = _MetaServerNode(self.worker_num, self.master_addr, self.master_port,
                              _generate_cmd_args_list(self.cmd, self.cmd_args),
                              os.path.join(self.log_dir, "scheduler.log"))
        self.msn_process = msn.run()

    def start_workers(self):
        """
        Starts the worker nodes.

        """
        for i in range(self.local_worker_num):
            node_id, log_name = self._get_node_id_and_log_path(i)
            cgn = _ComputeGraphNode(self.worker_num, self.master_addr, self.master_port,
                                    node_id, _generate_cmd_args_list(self.cmd, self.cmd_args), log_name)
            process = cgn.run()
            self.cgn_processes.append(process)
            if node_id is None:
                logger.warning(f"Rank ids will be assigned automatically, "
                               "please use 'grep -rn 'rank id:'' command to check each worker log's rank id.")
            else:
                logger.warning(f"Start worker process with rank id:{node_id}, log file:{log_name}")

    def heartbeat_with_scheduler(self):
        """
        Sends a heartbeat to the scheduler and updates the worker_num and local_worker_num.

        Returns:
            bool: True if the network has changed, False otherwise.

        """
        network_changed, worker_num, local_worker_num = _get_status_and_params(self.scheduler_url)
        self.worker_num = worker_num
        self.local_worker_num = local_worker_num
        return network_changed

    def join_processes(self):
        """
        Join all processes to stop.
        If there's any process does not exit normally, logs will be analyzed
        so that understandable root cause of exception could be returned.
        """
        has_exception = False
        for p in self.cgn_processes:
            p.wait()
            if p.returncode != 0:
                has_exception = True
                logger.error(f"Worker process {p.pid} exit with exception.")

        self.msn_process.wait()
        if self.msn_process.returncode != 0:
            has_exception = True
            logger.error(f"Scheduler process {self.msn_process.pid} exit with exception.")

        if has_exception:
            logger.warning("Analyzing exception log...")
            self._analyze_log()
            raise RuntimeError("Distributed job exited with exception. Please check logs and outputs.")

    def stop_processes(self):
        """
        Stops all running processes.

        """
        for p in self.cgn_processes:
            p.terminate()
            p.join()

        self.msn_process.terminate()
        self.msn_process.join()

    def stop_and_restart(self):
        """
        Stops all running processes and restarts the scheduler and workers.

        """
        self.stop_processes()
        if self.is_master:
            self.start_scheduler()
        self.start_workers()

    def _get_node_id_and_log_path(self, index):
        """
        Generate node id and log path for corresponding process.
        """
        if self.local_worker_num == self.worker_num:
            # This means only one node is involved.
            return index, os.path.join(self.log_dir, "worker_" + str(index) + ".log")

        if self.node_rank >= 0:
            # We assume that each node has same process number.
            node_id = self.node_rank * self.local_worker_num + index
            log_name = os.path.join(self.log_dir, "worker_" + str(node_id) + ".log")
        else:
            # If node_rank is default value -1, let MindSpore assign rank id.
            node_id = None
            log_name = os.path.join(self.log_dir, "worker_" + str(index) + ".log")
        return node_id, log_name

    def _analyze_log(self):
        """
        Analyze exception logs.
        """
        scheduler_log_path = os.path.join(self.log_dir, "scheduler.log")
        os.system(f"cat {scheduler_log_path}|grep -E 'ERROR|CRITICAL|Traceback' -C 10")
        time_out_node_ids = []
        with open(scheduler_log_path, "r") as log:
            scheduler_log = log.read()
            # Filter out abnormal logs.
            time_out_node_log = re.findall(r"Node [\s\S]* is timed out", scheduler_log)

            # Filter out node ids of the processes which exit abnormally.
            node_id_splitter = lambda l: re.split(" is timed out", re.split("Node ", l)[1])[0]
            time_out_node_ids = list(node_id_splitter(l) for l in time_out_node_log)

        # If 'time_out_node_ids' is not empty, only analyze logs of these time out nodes.
        # Unless get the error logs of all workers.
        if not time_out_node_ids:
            logger.error(f"Time out nodes are {time_out_node_ids}")
            # Get the logs which have these timeout node ids.
            grepper = lambda id: os.system(f"grep -rn 'node {id}' {self.log_dir}"
                                           "|awk -F: '{print $1}'")
            log_names = list(grepper(id) for id in time_out_node_ids)
            for log in log_names:
                os.system(f"cat {os.path.join(self.log_dir, log)}|grep -E 'ERROR|CRITICAL|Traceback' -C 10")
        else:
            os.system(f"grep -rn -E 'ERROR|CRITICAL|Traceback' -C 10 {self.log_dir}")

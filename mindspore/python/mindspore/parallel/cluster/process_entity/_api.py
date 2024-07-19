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
from ._utils import _generate_cmd_args_list, _generate_cmd_args_list_with_core, _generate_url,\
                    _is_local_ip, _send_scale_num

class _Node:
    """
    Base class for dynamic networking nodes.

    """
    def __init__(self, worker_num, sched_host, sched_port, timeout, args_list, output_file):
        self.worker_num = worker_num
        self.sched_host = sched_host
        self.sched_port = sched_port
        self.args_list = args_list
        self.output_file = output_file
        self.timeout = timeout

    def run(self):
        """
        Runs the node by setting environment variables and executing the entrypoint command or script.

        """
        os.environ["MS_WORKER_NUM"] = str(self.worker_num)
        os.environ["MS_SCHED_HOST"] = self.sched_host
        os.environ["MS_SCHED_PORT"] = str(self.sched_port)
        os.environ["MS_TOPO_TIMEOUT"] = str(self.timeout)

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
    def __init__(self, worker_num, sched_host, sched_port, timeout, node_id, args_list, output_file):
        super().__init__(worker_num, sched_host, sched_port, timeout, args_list, output_file)
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
        self.msn_process = None
        self.cgn_processes = []

        """`is_master` flags whether the current node is the master node."""
        self.is_master = _is_local_ip(args.master_addr)

        self.master_addr = args.master_addr
        self.master_port = args.master_port

        self.worker_num = args.worker_num
        if self.worker_num <= 0:
            raise ValueError(f"worker_num must be greater than 0, but got {self.worker_num}.")
        self.exported_rank_size = self.worker_num
        self.local_worker_num = args.local_worker_num
        self.node_rank = args.node_rank

        self.log_dir = args.log_dir
        self.join = args.join
        self.cluster_time_out = args.cluster_time_out
        self.bind_core = args.bind_core
        self.rank_table_file = args.rank_table_file

        self.sim_level = args.sim_level
        self.sim_rank_id = args.sim_rank_id
        self.is_simulation = (self.sim_level != -1)
        if self.is_simulation:
            # If simulation level is set, reset the worker_num and local_worker_num to 1
            # so that host cluster could be initialized.
            self.worker_num = 1
            self.local_worker_num = 1
            os.environ["MS_SIMULATION_LEVEL"] = str(self.sim_level)
        elif os.getenv("MS_SIMULATION_LEVEL"):
            # If simulation level env is set, load RANK_ID and RANK_SIZE envs.
            self.worker_num = 1
            self.local_worker_num = 1
            self.is_simulation = True
            self.sim_rank_id = os.getenv("RANK_ID", "0")
            if os.getenv("RANK_SIZE"):
                self.exported_rank_size = os.getenv("RANK_SIZE")

        self.cmd = args.task_script
        self.cmd_args = args.task_script_args

        """`is_scale` flags whether the current task is a scaling task and there is already a
        manager on the current node."""
        self.is_scale = False
        self.scheduler_url = _generate_url(self.master_addr, self.master_port)

        # Create log directory and set the permission if not exists.
        if self.log_dir and not os.path.exists(self.log_dir):
            permissions = os.R_OK | os.W_OK | os.X_OK
            origin_mask = os.umask(permissions << 3 | permissions)
            try:
                mode = permissions << 6
                os.makedirs(self.log_dir, mode=mode, exist_ok=True)
            finally:
                os.umask(origin_mask)

    def run(self):
        """
        Runs the process manager.

        """
        os.environ["RANK_SIZE"] = str(self.exported_rank_size)
        if self.rank_table_file != "":
            os.environ["RANK_TABLE_FILE"] = self.rank_table_file
            logger.warning(f"msrun launching distributed job with user configured rank table file path:"
                           f"{self.rank_table_file}")
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
        # For Scheduler, 'RANK_ID' is always 0.
        os.environ['RANK_ID'] = str(0)
        msn = _MetaServerNode(self.worker_num, self.master_addr, self.master_port, self.cluster_time_out,
                              _generate_cmd_args_list(self.cmd, self.cmd_args),
                              os.path.join(self.log_dir, "scheduler.log"))
        self.msn_process = msn.run()

    def start_workers(self):
        """
        Starts the worker nodes.

        """
        if self.local_worker_num == self.worker_num and self.node_rank not in [0, -1]:
            # If only one node is involved, ignore invalid 'node_rank'.
            logger.warning("All workers will be spawned on this node, "
                           f"so 'node_rank': [{self.node_rank}] will be ignored.")
        if self.local_worker_num < self.worker_num and self.node_rank == -1:
            logger.warning("You are running distributed job with multiple nodes but not setting '--node_rank'. So "
                           "'rank_id' of each process will be assigned after cluster is successfully built.\n"
                           "You can access 'RANK_ID' environment variable after calling "
                           "'mindspore.communication.init()'")

        if self.is_simulation and self.worker_num != 1:
            raise ValueError(f"Simulation level is set, worker_num must be 1, but got {self.worker_num}.")

        for i in range(self.local_worker_num):
            node_id, log_name = self._get_node_id_and_log_path(i)
            if node_id is None:
                logger.warning(f"Rank ids will be assigned automatically, "
                               "please use 'grep -rn 'rank id:' command to check each worker log's rank id.")
            else:
                # If node_id is generated in '_get_node_id_and_log_path' method, export 'RANK_ID' environment variable.
                # This is for rank_table method's compatibility consideration.
                os.environ["RANK_ID"] = str(node_id)
                logger.warning(f"Start worker process with rank id:{node_id}, log file:{log_name}. "
                               "Environment variable [RANK_ID] is exported.")
            if self.is_simulation:
                # Reset RANK_ID env to sim_rank_id.
                os.environ["RANK_ID"] = str(self.sim_rank_id)

            cpu_num = subprocess.getoutput("cat /proc/cpuinfo|grep processor|wc -l")
            if not cpu_num.isdigit():
                raise RuntimeError("Fail to get cpu number from /proc/cpuinfo.")
            if self.bind_core:
                avg = int(cpu_num) // self.local_worker_num
                cpu_start = avg * i
                cpu_end = cpu_start + avg - 1
                cmd = _generate_cmd_args_list_with_core(self.cmd, self.cmd_args, cpu_start, cpu_end)
            else:
                cmd = _generate_cmd_args_list(self.cmd, self.cmd_args)
            cgn = _ComputeGraphNode(self.worker_num, self.master_addr, self.master_port, self.cluster_time_out,
                                    node_id, cmd, log_name)
            process = cgn.run()
            self.cgn_processes.append(process)

    def join_processes(self):
        # pylint: disable=unused-variable
        """
        Join all processes to stop.
        If there's any process does not exit normally, logs will be analyzed
        so that understandable root cause of exception could be returned.
        """
        has_exception = False
        success_cgn_processes = set()
        while True:
            # Traversal all workers and kill immediately if any exception happens.
            for p in self.cgn_processes:
                ret_code = p.poll()
                if ret_code is None:
                    # This means the process is still running, poll next process.
                    continue
                elif ret_code != 0:
                    has_exception = True
                    logger.error(f"Worker process {p.pid} exit with exception.")
                    break
                else:
                    success_cgn_processes.add(p)

            if has_exception:
                logger.warning("There's worker exits with exception, kill all other workers.")
                for p in self.cgn_processes:
                    if p.poll() is None:
                        p.kill()
                break
            elif len(success_cgn_processes) == len(self.cgn_processes):
                logger.info("All workers successfully exit!")
                break


        if self.msn_process:
            self.msn_process.wait()
            if self.msn_process.returncode != 0:
                has_exception = True
                logger.error(f"Scheduler process {self.msn_process.pid} exit with exception.")

        if has_exception:
            logger.warning("Analyzing exception log...")
            self._analyze_log()
            raise RuntimeError("Distributed job exited with exception. Please check logs in "
                               f"directory: {self.log_dir}.")

    def stop_processes(self):
        """
        Stops all running processes.

        """
        for p in self.cgn_processes:
            p.terminate()
            p.join()

        if self.msn_process:
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
        if self.local_worker_num > self.worker_num:
            raise ValueError(f"Total worker number is {self.worker_num}, "
                             f"but got exceeded local worker number: {self.local_worker_num}.")
        if self.local_worker_num == self.worker_num:
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
        time_out_node_ids = []
        if os.path.exists(scheduler_log_path):
            with open(scheduler_log_path, "r") as log:
                scheduler_log = log.read()
                # Filter out abnormal logs.
                time_out_node_log = re.findall(r"node: .* is timed out", scheduler_log)

                # Filter out node ids of the processes which exit abnormally.
                def node_id_splitter(id):
                    return re.split(" is timed out", re.split("node: ", id)[1])[0]
                for id in time_out_node_log:
                    time_out_node_ids.append(node_id_splitter(id))
            logger.error(f"Time out nodes are {time_out_node_ids}")

        os.system(f"grep -rn -E 'ERROR|CRITICAL|Traceback|Error' -C 5 {self.log_dir}")

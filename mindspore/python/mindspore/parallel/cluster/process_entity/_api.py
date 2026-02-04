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
import time
import signal
import subprocess
import socket
import psutil
import mindspore.log as logger
from mindspore.utils import RSCPluginHandle
from ._utils import _generate_cmd, _generate_url, _is_local_ip, _convert_addr_to_ip, _send_scale_num, \
    _get_local_ip, _generate_auto_bind_core_strategy


class _Node:
    """
    Base class for dynamic networking nodes.

    """

    def __init__(self, worker_num, sched_host, sched_port, timeout, args_list, output_file, join, is_simulation):
        self.worker_num = worker_num
        self.sched_host = sched_host
        self.sched_port = sched_port
        self.args_list = args_list
        self.output_file = output_file
        self.timeout = timeout
        self.join = join
        self.is_simulation = is_simulation

    def run(self):
        """
        Runs the node by setting environment variables and executing the entrypoint command or script.

        """
        os.environ["MS_WORKER_NUM"] = str(self.worker_num)
        # If simulation level is set, environment variables for dynamic networking will not be set,
        # and scheduler will not be started.
        if not self.is_simulation:
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
        with open(self.output_file, "w", encoding='utf-8') as file_handle:
            return subprocess.Popen(self.args_list, stdout=file_handle, stderr=subprocess.STDOUT)


class _ComputeGraphNode(_Node):
    """
    Worker node for dynamic networking. Inherits from the Node class.
    """

    def __init__(self, worker_num, sched_host, sched_port, timeout, node_id, args_list, output_file,
                 tail_worker_log, join, is_simulation):
        super().__init__(worker_num, sched_host, sched_port, timeout, args_list, output_file, join, is_simulation)
        self.node_id = node_id
        self.tail_worker_log = tail_worker_log

    def run(self):
        """
        Runs the ComputeGraphNode by setting environment variables, setting the MS_NODE_ID variable
        to the node ID, setting the MS_ROLE variable to "MS_WORKER", and executing the entrypoint
        command or script.

        """
        super().run()
        if self.node_id is not None:
            os.environ["MS_NODE_ID"] = str(self.node_id)
        # If simulation level is set, environment variable 'MS_ROLE' will not be set.
        if not self.is_simulation:
            os.environ["MS_ROLE"] = "MS_WORKER"
        tail_worker_process = None
        is_tail_worker_log = self.enable_tail_worker_log()
        if self.join and not is_tail_worker_log:
            logger.warning(f"The '--tail_worker_log' is:{self.tail_worker_log}, "
                           f"which doesn't contain this worker {self.node_id}."
                           f" So this worker {self.node_id}'s log will not be output to console. Reset "
                           "'--tail_worker_log', if you want to output this worker's log to console.")
        with open(self.output_file, "w", encoding='utf-8') as file_handle:
            worker_process = subprocess.Popen(self.args_list, preexec_fn=os.setsid, stdout=file_handle,
                                              stderr=subprocess.STDOUT)
            if self.join and is_tail_worker_log:
                tail_worker_process = self.output_to_console()
            return worker_process, tail_worker_process

    def output_to_console(self):
        """
        Output worker log file to console.
        """
        return subprocess.Popen(['/usr/bin/tail', '-f', self.output_file])

    def enable_tail_worker_log(self):
        """
        Get valid rank ID for tailing the corresponding worker log.
        """
        tail_worker_log_list = []
        if self.tail_worker_log != "-1":
            tail_worker_log_list.extend([int(num) for num in self.tail_worker_log.split(',')])
        if self.tail_worker_log != "-1" and self.node_id not in tail_worker_log_list:
            return False
        return True


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
        self.tail_cgn_processes = []

        self.master_addr = _convert_addr_to_ip(args.master_addr)
        self.master_port = args.master_port

        """`is_master` flags whether the current node is the master node."""
        self.is_master = _is_local_ip(self.master_addr)

        self.worker_num = args.worker_num
        if self.worker_num <= 0:
            raise ValueError(f"worker_num must be greater than 0, but got {self.worker_num}.")
        self.exported_rank_size = self.worker_num
        self.local_worker_num = args.local_worker_num
        self.node_rank = args.node_rank

        self.log_dir = args.log_dir
        self.join = args.join
        self.worker_log_name = args.worker_log_name
        self.tail_worker_log = args.tail_worker_log
        self.cluster_time_out = args.cluster_time_out
        self.bind_core = args.bind_core
        self.rank_table_file = args.rank_table_file

        self.sim_level = args.sim_level
        self.sim_rank_id = args.sim_rank_id
        self.is_simulation = self.sim_level != -1
        if self.is_simulation:
            os.environ["MS_SIMULATION_LEVEL"] = str(self.sim_level)
            if self.sim_rank_id == -1:
                self.sim_rank_id = int(os.getenv("RANK_ID", "-1"))
        elif os.getenv("MS_SIMULATION_LEVEL"):
            self.is_simulation = True
            if self.sim_rank_id == -1:
                self.sim_rank_id = int(os.getenv("RANK_ID", "-1"))
            if os.getenv("RANK_SIZE"):
                self.exported_rank_size = os.getenv("RANK_SIZE")
        # If sim_rank_id is set, single worker can be started.
        if self.is_simulation and (self.sim_rank_id != -1):
            logger.info(f"Simulation rank id is set to {self.sim_rank_id}, will dryrun a single process.")
            self.local_worker_num = 1
        if self.is_simulation and self.local_worker_num > 128:
            self.local_worker_num = 1
            self.sim_rank_id = 0
            logger.warning("In dryrun case, local worker num is set to larger than 128. "
                           "To avoid a system clash, local worker num is set to 1.")

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

        self.device_to_cpu_map = {}
        if self.bind_core is True:
            self.device_to_cpu_map = _generate_auto_bind_core_strategy(self.local_worker_num)

        self.proc_rank_map = {}
        self.enable_mindx = False
        self.handler = None
        self._check_taskd()

    def _check_taskd(self):
        """check if enable taskd."""
        self.handler = RSCPluginHandle()
        self.enable_mindx = self.handler.check_enable()
        if self.enable_mindx is False:
            self.handler = None
            return
        ret = self.handler.register_callback({"KILL_WORKER": self.kill_workers,
                                              "START_ALL_WORKER": self.start_all_workers,
                                              "START_WORKER_LIST": self.start_worker_list,
                                              "MONITOR": self.monitor_rank_status
                                              })
        if not ret:
            logger.warning("Register callback to mindx failed, process controlled by msrun.")
            self.enable_mindx = False
            self.handler = None
            return
        logger.warning("Mindx enabled, process controlled by mindx.")
        os.environ["MS_ENABLE_RECOVERY"] = str(1)

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
            if self.is_master and not self.is_simulation:
                self.start_scheduler()
        if self.enable_mindx:
            self.handler.start()
        else:
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
        log_name = os.path.join(self.log_dir, "scheduler.log")
        cmd = _generate_cmd(self.cmd, self.cmd_args, -1, {}, self.bind_core)
        print(f"Start scheduler process, log file:{log_name}. Execute command: {' '.join(cmd)}", flush=True)

        msn = _MetaServerNode(self.worker_num, self.master_addr, self.master_port, self.cluster_time_out,
                              cmd, log_name, self.join, self.is_simulation)
        self.msn_process = msn.run()

    def _start_single_worker(self, local_rank):
        """
        Start worker processor

        Args:
            local_rank: local rank id.
        """
        os.environ["DEVICE_ID"] = str(local_rank)
        node_id, log_name = self._get_node_id_and_log_path(local_rank)
        if node_id is None:
            logger.warning("Rank ids will be assigned automatically, "
                           "please use 'grep -rn 'rank id:' command to check each worker log's rank id.")
        else:
            # If node_id is generated in '_get_node_id_and_log_path' method, export 'RANK_ID' environment variable.
            # This is for rank_table method's compatibility consideration.
            os.environ["RANK_ID"] = str(node_id)
        if self.is_simulation and (self.sim_rank_id != -1):
            # Reset RANK_ID env to sim_rank_id if sim_rank_id is set.
            os.environ["RANK_ID"] = str(self.sim_rank_id)
            logger.warning(f"In dryrun case, RANK_ID is assigned to {self.sim_rank_id}.")
        cmd = _generate_cmd(self.cmd, self.cmd_args, local_rank, self.device_to_cpu_map, self.bind_core)
        print(f"Start worker process with rank id:{node_id}, log file:{log_name}. "
              f"Environment variable [RANK_ID={node_id}] is exported. Execute command: {' '.join(cmd)}", flush=True)

        cgn = _ComputeGraphNode(self.worker_num, self.master_addr, self.master_port, self.cluster_time_out,
                                node_id, cmd, log_name, self.tail_worker_log, self.join, self.is_simulation)
        process, tail_process = cgn.run()
        self.cgn_processes.append(process)
        self.tail_cgn_processes.append(tail_process)
        self.proc_rank_map[local_rank] = process

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
        for i in range(self.local_worker_num):
            self._start_single_worker(i)

    def join_processes(self):
        """
        Join all processes to stop.
        If there's any process does not exit normally, logs will be analyzed
        so that understandable root cause of exception could be returned.
        """

        def signal_handler(sig, frame):
            logger.warning("msrun process received SIGNIN (Ctrl+C), terminating all workers.")
            self.kill_all_processes()
            sys.exit(0)

        has_exception = False
        success_cgn_processes = set()
        signal.signal(signal.SIGINT, signal_handler)
        while True:
            # Traversal all workers and kill immediately if any exception happens.
            for p in self.cgn_processes:
                ret_code = p.poll()
                if ret_code is None:
                    # This means the process is still running, poll next process.
                    continue
                if ret_code != 0:
                    has_exception = True
                    logger.error(f"Worker process {p.pid} exit with exception. Error code: {ret_code}.")
                    break
                success_cgn_processes.add(p)

            if has_exception:
                logger.warning("There's worker exits with exception, kill all other workers.")
                self.kill_worker_processes()
                self.kill_tail_log_processes()
                break
            if len(success_cgn_processes) == len(self.cgn_processes):
                logger.info("All workers successfully exit!")
                self.kill_tail_log_processes()
                break

            time.sleep(1)

        if self.msn_process:
            self.msn_process.wait()
            if self.msn_process.returncode != 0:
                has_exception = True
                logger.error(f"Scheduler process {self.msn_process.pid} exit with exception.")

        if has_exception:
            self._analyze_sched_log()
            raise RuntimeError("Distributed job exited with exception. Please check logs in "
                               f"directory: {self.log_dir}.")

    def kill_tail_log_processes(self):
        """
        Kills all tail worker log processes.

        """
        for p_tail in self.tail_cgn_processes:
            if p_tail is not None:
                logger.debug("Tail worker log process:{p_tail.pid} has been killed!")
                p_tail.kill()

    def kill_worker_processes(self):
        """
        Kills all worker processes.

        """
        for p in self.cgn_processes:
            if p.poll() is None:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)

    def kill_all_processes(self):
        """
        Kills all running processes, including scheduler, worker and tail log.

        """
        self.kill_worker_processes()
        self.kill_tail_log_processes()
        if self.msn_process.poll() is None:
            self.msn_process.kill()

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

    def kill_all_workers(self):
        """
        Kill all running worker processes.

        Args:
            NA.
        """
        self.kill_worker_processes()
        self.kill_tail_log_processes()
        self.cgn_processes.clear()
        self.tail_cgn_processes.clear()

    def kill_single_worker(self, pid):
        """
        Kill one worker process with specified pid.

        Args:
            pid: Worker process' pid.
        """
        kill_status = False
        for i in range(len(self.cgn_processes)):
            p = self.cgn_processes[i]
            if p.pid == pid and p.poll() is None:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                del self.cgn_processes[i]
                tail_p = self.tail_cgn_processes[i]
                if tail_p is not None:
                    tail_p.kill()
                del self.tail_cgn_processes[i]
                kill_status = True
                break
        if not kill_status:
            logger.warning(f"There's no active worker with pid: {pid}")

    def kill_workers(self, pids):
        """
        Kill worker process according to pids. Worker process with pid within pids list will be killed.

        Args:
            pids(list): a list of worker process pid. When local_ranks pids -1, kill all worker process.
        """
        if -1 in pids:
            self.kill_all_workers()
        else:
            for pid in pids:
                self.kill_single_worker(pid)
        return 0

    def monitor_rank_status(self, local_ranks):
        """
        Monitor the status of workers whose rank is within local_ranks list.

        Args:
            local_ranks(list): a list of local worker ranks. When local_ranks contains -1,
                monitor all workers' status.
        """
        rank_status = {}
        if -1 in local_ranks:
            local_ranks = list(range(self.local_worker_num))
        for i in local_ranks:
            single_status = self.monitor_single_rank(i)
            if single_status:
                rank_status[i] = single_status
        return rank_status

    def monitor_single_rank(self, rank_id):
        """
        Monitor the status of a single worker with rank_id

        Args:
            rank_id: worker process's local rank, which is also device_id.
        """
        if 0 <= rank_id < self.local_worker_num:
            global_rank_id = rank_id
            if self.node_rank >= 0:
                global_rank_id = self.node_rank * self.local_worker_num + rank_id
            try:
                p = self.proc_rank_map[rank_id]
                p_status = p.poll()
                if (not psutil.pid_exists(p.pid)) and (p_status != 0):
                    p_status = 300
                return {"pid": p.pid, "status": p_status, "global_rank": global_rank_id, "local_rank": rank_id,
                        "node_id": self.node_rank}
            except KeyError:
                logger.info(f"Process rank {rank_id} has not been initialized.")
                return {"pid": None, "status": 200, "global_rank": global_rank_id}
        else:
            logger.warning("Invalid rank id!")
        return {}

    def start_all_workers(self):
        """
        Start all worker processes after killing all workers.

        Args:
            NA.
        """
        if self.cgn_processes:
            self.kill_all_workers()
        self.start_workers()
        worker_status = self.monitor_rank_status([-1])
        for i in range(self.local_worker_num):
            if worker_status[i]["status"] is not None:
                return 1
        return 0

    def start_worker_list(self, rank_ids):
        """
        Start worker processor by rank list.

        Args:
            rank_ids: worker process's local rank list, which is also device_id.
        """
        if not isinstance(rank_ids, list):
            raise TypeError(f"The type of 'rank_ids' must be a list, but got:{rank_ids}")
        for idx in rank_ids:
            self._start_single_worker(idx)
        worker_status = self.monitor_rank_status(rank_ids)
        for i in rank_ids:
            if worker_status[i]["status"] is not None:
                return 1
        return 0

    def _get_node_id_and_log_path(self, index):
        """
        Generate node id and log path for corresponding process.
        """
        formatted_log_name = self.format_worker_log_name()
        if self.local_worker_num > self.worker_num:
            raise ValueError(f"Total worker number is {self.worker_num}, "
                             f"but got exceeded local worker number: {self.local_worker_num}.")
        if self.local_worker_num == self.worker_num:
            return index, os.path.join(self.log_dir, formatted_log_name + "_" + str(index) + ".log")

        if self.node_rank >= 0:
            # We assume that each node has same process number.
            node_id = self.node_rank * self.local_worker_num + index
            log_name = os.path.join(self.log_dir, formatted_log_name + "_" + str(node_id) + ".log")
        else:
            # If node_rank is default value -1, let MindSpore assign rank id.
            node_id = None
            log_name = os.path.join(self.log_dir, formatted_log_name + "_" + str(index) + ".log")
        return node_id, log_name

    def _analyze_sched_log(self):
        """
        Analyze scheduler log.
        """
        scheduler_log_path = os.path.join(self.log_dir, "scheduler.log")
        if os.path.exists(scheduler_log_path):
            os.system(f"cat {scheduler_log_path} | grep -E 'ERROR|CRITICAL|Traceback|Error' -C 5")

    def format_worker_log_name(self):
        """
        Format worker log files' name.
        """
        if not self.worker_log_name:
            formatted_worker_log_name = "worker"
        else:
            current_ip = _get_local_ip(self.master_addr)
            formatted_worker_log_name = re.sub(r'\{ip\}', current_ip, self.worker_log_name)
            formatted_worker_log_name = re.sub(r'\{hostname\}', socket.gethostname(), formatted_worker_log_name)
        return formatted_worker_log_name

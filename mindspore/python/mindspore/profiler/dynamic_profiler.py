# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
"""Dynamic Profile Monitor"""
import os
import sys
import json
import time
import stat
import atexit
import random
import multiprocessing

from mindspore import log as logger
from mindspore.train import Callback
from mindspore.profiler import tensorboard_trace_handler, schedule
from mindspore.profiler.profiler import Profile
from mindspore.profiler.experimental_config import _ExperimentalConfig
from mindspore.profiler.common.file_manager import FileManager
from mindspore.profiler.common.path_manager import PathManager
from mindspore.profiler.dynamic_profile.dynamic_profiler_config_context import DynamicProfilerConfigContext
from mindspore.profiler.dynamic_profile.dynamic_monitor_proxy import MsDynamicMonitorProxySingleton
from mindspore.profiler.dynamic_profile.dynamic_profiler_utils import DynamicProfilerUtils
from mindspore.profiler.common.util import no_exception_func
from mindspore.profiler.profiler_interface import ProfilerInterface


def print_msg(msg):
    """print msg"""
    print("[Dynamic Profiler] " + msg, flush=True)


class DynamicProfilerMonitorBase(Callback):
    """
    Dynamic profiler callback base class implementing the dynamic profiler functionality.
    """

    NPU_MONITOR_START = "NPU_MONITOR_START"

    def __init__(self, cfg_path=None, output_path=None, poll_interval=2, **kwargs):
        self._is_dyno = DynamicProfilerUtils.is_dyno_mode()
        self._rank_id = DynamicProfilerUtils.get_real_rank()
        if not self._is_dyno:
            self._cfg_path = cfg_path
            self._cfg_json_path = os.path.join(self._cfg_path, "profiler_config.json")
            self._cfg_json_path = os.path.realpath(self._cfg_json_path)
            self._init_cfg_json()
        self._output_path = "dyn_profile_data" if output_path is None else output_path
        self._poll_interval = poll_interval
        if not isinstance(self._poll_interval, int):
            logger.error("Poll interval must be an integer, reset to 2.")
            self._poll_interval = 2

        if self._poll_interval < 1:
            logger.error("Poll interval must be greater than 1, reset to 2.")
            self._poll_interval = 2

        self._kwargs = kwargs
        self._shm_name = time.strftime("DynamicProfileShm%Y%m%d%H", time.localtime())
        self._shared_loop_flag = multiprocessing.Value('b', True)
        self._profiler_status = {
            DynamicProfilerUtils.PROFILER_STATUS: str(DynamicProfilerUtils.ProfilerStatus.IDLE.value),
            DynamicProfilerUtils.CURRENT_STEP: "-1",
            DynamicProfilerUtils.START_STEP: "-1",
            DynamicProfilerUtils.STOP_STEP: "-1",
        }
        self._last_report_time = 0.0
        self._shm = None
        self._process = None
        self._profiler = None
        self._last_start_step = None
        self._last_stop_step = None
        self._is_create_process = None
        self._is_started = False
        self._start_step = -1
        self._stop_step = -1
        self._step_num = 0
        self._collection_step_num = 0
        self._check_shm_for_killed()
        self._create_shm()
        self._create_process()
        atexit.register(self._clean_resource)
        if self._is_dyno:
            atexit.register(self._finalize_dynolog)

    @no_exception_func()
    def step_begin(self, run_context):
        """
        Start profiler at the begin of step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        prof_json = self._get_prof_args()
        if not prof_json:
            return
        if self._is_dyno:
            # Dyno monitor process
            if self.NPU_MONITOR_START in prof_json:
                self._call_dyno_monitor(prof_json)
                return

        prof_args = DynamicProfilerConfigContext(prof_json)
        if not prof_args.is_valid:
            logger.error("Dynamic profiler json is not valid, please check the json file.")
            return

        if prof_args.start_step in (-1, self._last_start_step):
            return

        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        start_step, stop_step = self._check_step(prof_args.start_step, prof_args.stop_step, step_num)

        # Prevent repeated calls of the start function within a complete interval
        if step_num == start_step:
            if self._is_started:
                logger.error("Dynamic profiler is already started at step %d, "
                             "please wait the first profiler finished at step %d.",
                             self._last_start_step, self._last_stop_step)
                return

            if self._profiler is None:
                output_path = prof_args.prof_path if prof_args.prof_path != "./" else self._output_path
                prof_path = os.path.join(
                    output_path,
                    f"rank{self._rank_id}_start{start_step}_stop{stop_step}"
                )
                PathManager.check_input_directory_path(prof_path)
                profiler_config = self._get_prof_config(prof_args, prof_path, start_step, stop_step,
                                                        start_profile=False,
                                                        skip_first=0)
                self._profiler = Profile(**profiler_config)
                print_msg(f"Rank {self._rank_id} create output path {prof_path}")

            self._profiler.start()
            self._is_started = True
            self._last_start_step = start_step
            self._last_stop_step = stop_step
            print_msg(f"Rank {self._rank_id} Dynamic profiler start at step {start_step}, "
                      f"will stop at step {stop_step}")

    @staticmethod
    def _get_prof_config(prof_args, prof_path, start_step, stop_step, start_profile, skip_first):
        """
        Get profiler config.

        Args:
            prof_args: Profiler config.
            prof_path: Profiler output path.
            start_step: Start step.
            stop_step: Stop step.
            start_profile: enable start_profile.
            skip_first: skip first step.
        """
        profiler_config = {
            "activities": prof_args.args.get("activities"),
            "with_stack": prof_args.args.get("with_stack"),
            "profile_memory": prof_args.args.get("profile_memory"),
            "parallel_strategy": prof_args.args.get("parallel_strategy"),
            "start_profile": start_profile,
            "record_shapes": prof_args.args.get("record_shapes"),
            "schedule": schedule(
                wait=0,
                warmup=0,
                active=stop_step - start_step + 1,
                repeat=1,
                skip_first=skip_first
            ),
            "on_trace_ready": tensorboard_trace_handler(
                dir_name=prof_path,
                analyse_flag=prof_args.analyse,
                async_mode=prof_args.analyse_mode == "async",
            ),
            "experimental_config": _ExperimentalConfig(
                profiler_level=prof_args.args.get("profiler_level"),
                aic_metrics=prof_args.args.get("aic_metrics"),
                l2_cache=prof_args.args.get("l2_cache"),
                mstx=prof_args.args.get("mstx"),
                data_simplification=prof_args.args.get("data_simplification"),
                export_type=prof_args.args.get("export_type"),
                mstx_domain_include=prof_args.args.get("mstx_domain_include"),
                mstx_domain_exclude=prof_args.args.get("mstx_domain_exclude"),
                sys_io=prof_args.args.get("sys_io"),
                sys_interconnection=prof_args.args.get("sys_interconnection"),
                host_sys=prof_args.args.get("host_sys")
            )
        }
        return profiler_config

    @no_exception_func()
    def step_end(self, run_context):
        """
        Stop profiler at the end of step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        prof_json = self._get_prof_args()
        prof_args = DynamicProfilerConfigContext(prof_json)

        if not prof_args.is_valid:
            logger.error("Dynamic profiler json is not valid, please check the json file.")
            return

        if prof_args.stop_step == -1:
            return

        if self._profiler:
            self._profiler.step()

        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num

        if step_num == self._last_stop_step and self._is_started:
            self._profiler = None
            self._is_started = False
            print_msg(f"Rank {self._rank_id} Dynamic profiler stop at step {step_num}")

    @no_exception_func()
    def step(self):
        """
        Used for Ascend, distinguish step collection and parsing performance data by dynamic profiler.

        Raises:
            RuntimeError: If the 'start_step' parameter setting is greater than the 'stop_step' parameter setting.

        Examples:
            >>> import json
            >>> import os
            >>> import numpy as np
            >>>
            >>> import mindspore
            >>> import mindspore.dataset as ds
            >>> from mindspore import context, nn
            >>> from mindspore.profiler import DynamicProfilerMonitor
            >>>
            >>>
            >>> class Net(nn.Cell):
            ...     def __init__(self):
            ...         super(Net, self).__init__()
            ...         self.fc = nn.Dense(2, 2)
            ...
            ...     def construct(self, x):
            ...         return self.fc(x)
            >>>
            >>> def generator_net():
            ...     for _ in range(2):
            ...         yield np.ones([2, 2]).astype(np.float32), np.ones([2]).astype(np.int32)
            >>>
            >>> def train(test_net):
            ...     optimizer = nn.Momentum(test_net.trainable_params(), 1, 0.9)
            ...     loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
            ...     data = ds.GeneratorDataset(generator_net(), ["data", "label"])
            ...     model = mindspore.train.Model(test_net, loss, optimizer)
            ...     model.train(1, data)
            >>>
            >>> def change_cfg_json(json_path):
            ...     with open(json_path, 'r', encoding='utf-8') as file:
            ...          data = json.load(file)
            ...
            ...     data['start_step'] = 6
            ...     data['stop_step'] = 7
            ...
            ...     with open(json_path, 'w', encoding='utf-8') as file:
            ...          json.dump(data, file, ensure_ascii=False, indent=4)
            >>>
            >>> if __name__ == '__main__':
            ...      # set json configuration file
            ...      context.set_context(mode=mindspore.PYNATIVE_MODE)
            ...      mindspore.set_device("Ascend")
            ...      data_cfg = {
            ...           "start_step": 2,
            ...           "stop_step": 5,
            ...           "aic_metrics": "AiCoreNone",
            ...           "profiler_level": "Level0",
            ...           "analyse_mode": 0,
            ...           "activities": ["CPU", "NPU"],
            ...           "export_type": ["text"],
            ...           "profile_memory": False,
            ...           "mstx": False,
            ...           "parallel_strategy": False,
            ...           "with_stack": False,
            ...           "data_simplification": True,
            ...           "l2_cache": False,
            ...           "analyse": True,
            ...           "record_shape": False,
            ...           "prof_path": "./data",
            ...           "mstx_domain_include": [],
            ...           "mstx_domain_exclude": [],
            ...           "host_sys": [],
            ...           "sys_io": False,
            ...           "sys_interconnection": False
            ...      }
            ...      output_path = "./cfg_path"
            ...      cfg_path = os.path.join(output_path, "profiler_config.json")
            ...      os.makedirs(output_path, exist_ok=True)
            ...      # set cfg file
            ...      with open(cfg_path, 'w') as f:
            ...           json.dump(data_cfg, f, indent=4)
            ...      # cfg_path contains the json configuration file path, and output_path is the output path
            ...      dp = DynamicProfilerMonitor(cfg_path=output_path, output_path=output_path)
            ...      STEP_NUM = 15
            ...      # Define a network of training models
            ...      net = Net()
            ...      for i in range(STEP_NUM):
            ...          print(f"step {i}")
            ...          train(net)
            ...          # Modify the configuration file after step 7
            ...          # For example, change start_step to 8 and stop_step to 10
            ...          if i == 5:
            ...             # Modify parameters in the JSON file
            ...             change_cfg_json(os.path.join(output_path, "profiler_config.json"))
            ...          # Call step collection
            ...          dp.step()
        """
        self._step_num += 1
        prof_json = self._get_prof_args()
        if not prof_json:
            self._update_step_info(self._step_num, DynamicProfilerUtils.CURRENT_STEP)
            self._report_profiler_status()
            return
        if self._is_dyno:
            # Dyno monitor process
            if self.NPU_MONITOR_START in prof_json:
                self._call_dyno_monitor(prof_json)
                self._update_step_info(self._step_num, DynamicProfilerUtils.CURRENT_STEP)
                self._report_profiler_status()
                return

        prof_args = DynamicProfilerConfigContext(prof_json)
        if not prof_args.is_valid:
            logger.error("Dynamic profiler config is not valid, please check the json or dyno config.")
            self._update_step_info(self._step_num, DynamicProfilerUtils.CURRENT_STEP)
            self._report_profiler_status()
            return
        self._handle_profiler_setup(prof_args)

        if self._profiler:
            self._profiler.step()
            self._collection_step_num -= 1
            if self._collection_step_num == -1:
                self._profiler = None
                self._update_profiler_status(DynamicProfilerUtils.ProfilerStatus.IDLE)
        self._update_step_info(self._step_num, DynamicProfilerUtils.CURRENT_STEP)
        self._report_profiler_status()

    def _handle_profiler_setup(self, args):
        """Common handler for profiler setup logic shared between dyno and non-dyno paths."""
        start_step = args.start_step
        stop_step = args.stop_step

        if not self._is_valid_start_stop_step(self._step_num, start_step, stop_step):
            return

        if self._start_step != start_step or self._stop_step != stop_step:
            self._start_step = start_step
            self._stop_step = stop_step

            if not (start_step >= 0 and 0 <= start_step <= stop_step):
                self._profiler = None
                logger.error(
                    "Rank %d Dynamic profiler start at step %d and stop at step %d must be "
                    "greater than or equal to 0, and stop step should not be less than start step",
                    self._rank_id, start_step, stop_step
                )
                return

            # Setup profiler configuration
            output_path = args.prof_path if args.prof_path != "./" else self._output_path
            prof_path = os.path.join(
                output_path,
                f"rank{self._rank_id}_start{start_step}_stop{stop_step}"
            )
            print_msg(f"Rank {self._rank_id} create output path {prof_path}")
            print_msg(
                f"Rank {self._rank_id} Dynamic profiler start at step {start_step}, "
                f"will stop at step {stop_step}"
            )
            profiler_config = self._get_prof_config(args, prof_path, start_step, stop_step, start_profile=True,
                                                    skip_first=1)
            self._profiler = Profile(**profiler_config)
            self._collection_step_num = stop_step - start_step + 1
            self._update_step_info(start_step, DynamicProfilerUtils.START_STEP)
            self._update_step_info(stop_step, DynamicProfilerUtils.STOP_STEP)
            self._update_profiler_status(DynamicProfilerUtils.ProfilerStatus.RUNNING)

    def _is_valid_start_stop_step(self, step_num, start_step, stop_step):
        """Verify whether start_step and stop_step are valid parameters."""
        if start_step < 0 or stop_step < 0:
            return False

        if step_num < start_step:
            return False

        if step_num > stop_step != self._stop_step:
            logger.warning("stop_step must be greater than step_num, "
                           "but get start_step = %d, stop_step = %d, step_num = %d", start_step, stop_step, step_num)
            return False

        return True

    def _update_profiler_status(self, status: DynamicProfilerUtils.ProfilerStatus):
        self._profiler_status[DynamicProfilerUtils.PROFILER_STATUS] = str(status.value)

    def _update_step_info(self, step: int, flag: str):
        if flag in self._profiler_status:
            self._profiler_status[flag] = str(step)

    def _report_profiler_status(self):
        """Report profiler status"""
        if not self._is_create_process:
            return
        current_time = time.time()
        if current_time - self._last_report_time < DynamicProfilerUtils.REPORT_INTERVAL:
            return
        dyno_monitor_proxy = MsDynamicMonitorProxySingleton().get_proxy()
        if not dyno_monitor_proxy or not hasattr(dyno_monitor_proxy, "update_profiler_status"):
            return
        dyno_monitor_proxy.update_profiler_status(self._profiler_status)
        self._last_report_time = current_time

    @no_exception_func()
    def _call_dyno_monitor(self, dyno_args):
        if "is_valid" in dyno_args:
            del dyno_args["is_valid"]
        dyno_monitor_proxy = MsDynamicMonitorProxySingleton().get_proxy()
        dyno_monitor_proxy.enable_dyno_npu_monitor(dyno_args)

    @no_exception_func()
    def on_train_end(self, run_context):
        """
        Callback on trian end

        Args:
            run_context (RunContext): Context of the train running.
        """
        self._clean_resource()

    def _get_prof_args(self):
        """ Get prof_args """
        logger.error("Dynamic profiler _get_prof_args is not implemented")
        return {}

    def _clean_resource(self):
        """Clean resource"""
        logger.error("Dynamic profiler _clean_resource is not implemented")

    def _finalize_dynolog(self):
        """finalize dynolog"""
        logger.error("Dynolog monitor _finalize_dynolog is not implemented")

    def _check_step(self, start_step, stop_step, step_num):
        """Check step valid"""
        if start_step <= 0 or stop_step <= 0:
            return -1, -1

        if start_step > stop_step:
            logger.error("start_step must be less than stop_step, "
                         "but get start_step = %d, stop_step = %d", start_step, stop_step)
            return -1, -1

        if start_step < step_num and start_step != self._last_start_step:
            logger.error("start_step must be greater than step_num, "
                         "but get start_step = %d, stop_step = %d, step_num = %d", start_step, stop_step, step_num)
            return -1, -1

        if stop_step < step_num and stop_step != self._last_stop_step:
            logger.error("stop_step must be greater than step_num, "
                         "but get start_step = %d, stop_step = %d, step_num = %d", start_step, stop_step, step_num)
            return -1, -1

        return start_step, stop_step

    @no_exception_func()
    def _init_cfg_json(self):
        """Init config json file"""
        if self._rank_id == 0:
            if not os.path.exists(self._cfg_json_path):
                logger.info("cfg_path is not exist, create default cfg json")
                default_dy_config_context = DynamicProfilerConfigContext({})
                PathManager.make_dir_safety(self._cfg_path)
                config_file_path = os.path.join(self._cfg_path, "profiler_config.json")
                FileManager.create_json_file(config_file_path, default_dy_config_context.vars, indent=4)
        else:
            logger.info("rank_id is not 0, skip init cfg json")
        print_msg(f"Init config json file: {self._cfg_json_path}")

    def _create_shm(self):
        """Create a json monitor process based on whether the SharedMemory is successfully created"""
        logger.error("Dynamic profiler _create_shm is not implemented")

    @no_exception_func()
    def _create_process(self):
        """Create json monitor process, one process will be created at one worker"""
        if self._is_create_process:
            args = [self._shared_loop_flag, self._poll_interval, self._shm, self._rank_id] \
                if self._is_dyno else \
                [self._shared_loop_flag, self._poll_interval, self._shm, self._cfg_json_path]
            # daemon need to be set to True, otherwise the process will not be killed when the main process exits.
            ctx = multiprocessing.get_context("fork")
            self._process = ctx.Process(target=worker_dyno_func if self._is_dyno else worker_func,
                                        daemon=True,
                                        args=args)
            self._process.start()
            logger.info("Config monitor process has been created by rank %d.", self._rank_id)
        else:
            self._process = None
            logger.info("Rank %d no need to create process.", self._rank_id)

    @no_exception_func()
    def _check_shm_for_killed(self):
        """
        User killed process shm can not clean normally, so check this when create shm.
        """
        if sys.version_info >= (3, 8):
            shm_path = os.path.join("/dev/shm", self._shm_name)
        else:
            shm_path = self._shm_path

        if not os.path.exists(shm_path):
            return

        max_time_diff = 900  # seconds
        time_shm = os.stat(shm_path).st_ctime
        cur_proc_time = self._get_pid_st_ctime(os.getpid())

        if cur_proc_time and abs(cur_proc_time - time_shm) > max_time_diff:
            logger.error("There maybe exist share memory before this task, if you kill last task, "
                         "dynamic profiler will not valid, please remove %s, and retry." % shm_path)
            return

    def _get_pid_st_ctime(self, pid):
        """Get pid st_ctime"""
        create_time = 0.0
        try:
            fd = os.open(os.path.join('/proc', str(pid)), os.O_RDONLY, stat.S_IRUSR | stat.S_IRGRP)
            stat_ino = os.fstat(fd)
            os.close(fd)
            create_time = stat_ino.st_ctime
            return create_time
        except FileNotFoundError:
            logger.error("Process with PID %d does not exist.", pid)
        except PermissionError:
            logger.error("Permission denied when accessing PID %d.", pid)
        except Exception as ex:  # pylint: disable=W0703
            logger.error("An error occurred while getting creation time for PID %d: %s", pid, str(ex))
        return create_time


if sys.version_info >= (3, 8):
    @no_exception_func()
    def write_bytes(shm, byte_data):
        """Write bytes to shared memory"""
        shm.buf[:] = b'\x00' * len(shm.buf)
        shm.buf[:len(byte_data)] = byte_data
else:
    @no_exception_func()
    def write_bytes(shm, byte_data):
        """Write bytes to shared memory"""
        shm.seek(0)
        shm.write(byte_data)


@no_exception_func()
def worker_func(loop_flag, poll_interval, shm, cfg_path):
    """ Json monitor process worker function python version >= 3.8"""
    last_file_t = None
    while loop_flag.value:
        if os.path.exists(cfg_path):
            file_t = os.path.getmtime(cfg_path)
            if not last_file_t or last_file_t != file_t:
                last_file_t = file_t

                try:
                    data = FileManager.read_json_file(cfg_path)
                    data['is_valid'] = True
                    logger.info("Dynamic profiler process load json success")
                except json.JSONDecodeError as e:
                    data = {'is_valid': False}
                    logger.error("Dynamic profiler process load json failed: %s", e)
                # convert json to bytes
                byte_data = DynamicProfilerConfigContext.json_to_bytes(data)
                write_bytes(shm, byte_data)
        else:
            logger.error("Dynamic profiler cfg json not exists")
        time.sleep(poll_interval)
    logger.info("Dynamic profiler process done")


@no_exception_func()
def worker_dyno_func(loop_flag, poll_interval, shm, rank_id):
    """ dyno monitor process worker function python version >= 3.8"""
    proxy = MsDynamicMonitorProxySingleton().get_proxy()
    ret = proxy.init_dyno(rank_id)

    if not ret:
        logger.warning("Rank %d init dynolog failed !")
        return
    print_msg("Init dynolog success !")

    while loop_flag.value:
        try:
            res = proxy.poll_dyno()
            if not res:
                continue
            data = DynamicProfilerUtils.dyno_str_to_dict(res)
        except Exception as e:  # pylint: disable=broad-except
            data = {'is_valid': False}
            logger.error("Dynolog process load config failed: %s", e)
        else:
            data['is_valid'] = True

        # convert dyno config json to bytes
        byte_data = DynamicProfilerConfigContext.json_to_bytes(data)
        write_bytes(shm, byte_data)
        time.sleep(poll_interval)
    if hasattr(proxy, "update_profiler_status"):
        profiler_status = {
            DynamicProfilerUtils.PROFILER_STATUS: str(DynamicProfilerUtils.ProfilerStatus.UNINITIALIZED.value),
            DynamicProfilerUtils.CURRENT_STEP: "-1",
            DynamicProfilerUtils.START_STEP: "-1",
            DynamicProfilerUtils.STOP_STEP: "-1",
        }
        proxy.update_profiler_status(profiler_status)
    logger.info("Dynolog process done")


if sys.version_info >= (3, 8):
    from multiprocessing import shared_memory
    from unittest.mock import patch


    class DynamicProfilerMonitor(DynamicProfilerMonitorBase):
        r"""
        This class to enable the dynamic profiler monitoring of MindSpore neural networks.

        Args:
            cfg_path (str): (Ascend only) Dynamic profiler json config file directory. The requirement is a shared path
                that can be accessed by all nodes. The parameters of the json configuration file are as follows:

                - start_step (int, required) - Sets the step number at which the Profiler starts collecting data.
                  It is a relative value, with the first step of training being 1. The default value is -1, indicating
                  that data collection will not start during the entire training process.
                - stop_step (int, required) - Sets the step number at which the Profiler stops collecting data. It is
                  a relative value, with the first step of training being 1. The stop_step must be greater than or
                  equal to start_step. The default value is -1, indicating that data collection will not start during
                  the entire training process.
                - aic_metrics (int/str, optional) - Set the collection of AI Core metric data. The current version can
                  pass in either type int or str. Later, it will be updated to only pass in the str type.
                  Here, ``0`` and ``"PipeUtilization"`` represent PipeUtilization; ``1`` and ``"ArithmeticUtilization"``
                  represent ArithmeticUtilization; ``2`` and ``"Memory"`` represent Memory; ``3`` and ``"MemoryL0"``
                  represent MemoryL0; ``4`` and ``"MemoryUB"`` stand for MemoryUB; ``5`` and ``"ResourceConflictRatio"``
                  represent ResourceConflictRatio; ``6`` and ``"L2Cache"`` represent L2Cache; ``7`` and
                  ``"MemoryAccess"`` stand for MemoryAccess. The default value ``"AiCoreNone"`` indicates that the
                  AI Core metric is not collected.
                - profiler_level (int/str, optional) - Set the level for collecting performance data. The current
                  version can pass in either type int or str, and it will be updated to only pass in str type
                  in the future. Among them, ``-1`` and ``"LevelNone"`` represent ProfilerLevel.LevelNone, ``0``
                  and ``"Level0"`` represent ProfilerLevel.Level0, and ``1`` and ``"Level1"`` represent
                  ProfilerLevel.Level1. ``2`` and ``"Level2"`` stand for Profile Level.Level2.
                  The default value ``"Level0"`` indicates the collection level of ProfilerLevel.Level0.
                - activities (int/list, optional) - Set the device for collecting performance data.
                  The current version can pass in either type int or list. Later, it will be updated to only
                  pass in the list type. Among them, ``0`` and ``["CPU","NPU"]`` represent CPU+NPU, ``1`` and
                  ``["CPU"]`` represent CPU, and ``2`` and ``["NPU"]`` represent NPU. The default values
                  ``["CPU","NPU"]`` indicate the collection of  performance data of CPU+NPU.
                - export_type (int/list, optional) - Set the type of the exported performance data.
                  The current version can pass in either type int or list, and it will be updated later
                  to only pass in the list type. Among them, ``0`` and ``["text"]`` represent text, ``1`` and ``["db"]``
                  represent db, and ``2`` and ``["text","db"]`` represent text and db respectively. The default value
                  ``["text"]`` indicates that only performance data of the text type is exported.
                - profile_memory (bool, optional) - Set whether to collect memory performance data, true indicates that
                  memory performance data is collected, false indicates that memory performance data is not collected.
                  The default value is false, indicating that memory performance data is not collected.
                - mstx (bool, optional) - Set whether to enable mstx, true indicates that mstx is enabled, false
                  indicates that mstx is disabled. The default value is false, indicating that mstx is not enabled.
                - analyse (bool, optional) - Set whether to enable online analysis. True indicates that online analysis
                  is enabled, while false indicates that online analysis is disabled. The default value is false,
                  indicating that online analysis is not enabled. This parameter has a higher priority than the
                  `analyse_mode` parameter. When this parameter is set to false, the setting of the `analyse_mode`
                  parameter does not take effect. When this parameter is set to true,
                  setting the `analyse_mode` parameter to -1 does not take effect.
                - analyse_mode (int, optional) - Sets the mode for online analysis,
                  where 0 represents "sync" and 1 represents "async". The default value is -1,
                  indicating that online analysis is not used. This parameter has a lower priority than the `analyse`
                  parameter. When the `analyse` parameter is set to false, the setting of this parameter does not take
                  effect. When the `analyse` parameter is set to true, setting it to -1 does not take effect.
                - parallel_strategy (bool, optional) - Sets whether to collect parallel strategy performance data,
                  where true means to collect and false means not to collect. The default value is false, indicating
                  that parallel strategy performance data is not collected.
                - with_stack (bool, optional) - Sets whether to collect call stack information, where true means to
                  collect and false means not to collect. The default value is false, indicating that call stack
                  information is not collected.
                - data_simplification (bool, optional) - Sets whether to enable data simplification, where true means
                  to enable and false means not to enable. The default value is true, indicating that data
                  simplification is enabled.
                - record_shapes (bool, optional) - Sets whether to collect operator input tensor shapes data, where true
                  means that the shape data is collected and false means that the shape data is not collected. The
                  default value is false, indicating that input tensor shapes data is not collected.
                - mstx_domain_include (list, optional) - Set the set of enabled domain names when the mstx switch
                  is turned on. The name must be of str type. Default value: ``[]``, indicating that this parameter
                  is not used to control the domain. This parameter is mutually exclusive with the mstx_domain_exclude
                  parameter and cannot be set. simultaneously. If both are set, only the mstx_domain_include parameter
                  takes effect.
                - mstx_domain_exclude (list, optional) - Set the set of domain names that are not enabled when the
                  mstx switch is turned on. The name must be of str type. Default value: ``[]``, indicating that this
                  parameter is not used to control the domain.
                - prof_path (str, optional) - Output data path of the dynamic profiler. It is the same as the interface
                  parameter `output_path`. When both are set, `prof_path` takes effect. Default value:
                  ``"./"`` .
                - sys_io (bool, optional) - Set whether to collect NIC and RoCE data. Default value: ``False`` ,
                  indicating that these data are not collected.
                - sys_interconnection (bool, optional) - Set whether to collect system interconnection data,
                  including aggregate collective communication statistics (HCCS), PCIe data, and inter-chip transmission
                  bandwidth information. Default value: ``False`` , indicating that these data are not collected.
                - host_sys (list, optional) - Collect the data of system class calls, storage classes and cpu usage
                  rate on the host side, and pass in the list type. It supports passing in one or more of ``"cpu"``,
                  ``"mem"``, ``"disk"``, ``"network"`` and ``"osrt"``. Among them, ``"cpu"`` represents the cpu
                  utilization at the process level, ``"mem"`` represents the memory utilization at the process level,
                  ``"disk"`` represents the disk I/O utilization at the process level, and ``"network"`` represents the
                  network I/O utilization at the system level. ``"osrt"`` represents system-level syscall and
                  pthreadcall. Default value: ``[]``, indicating that system class data on the host side is
                  not collected. When collecting DISK or OSRT data, it is necessary to install the iotop, perf,
                  and ltrace third-party tools in advance. For detailed steps, please refer to
                  `Installing Third-party Tools <https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/T&ITools/
                  Profiling/atlasprofiling_16_0136.html>`_ .
                  After the third-party tool is successfully installed, user permissions need to be configured. For
                  detailed steps, please refer to `Configure User Permissions <https://www.hiascend.com/document/
                  detail/zh/mindstudio/80RC1/T&ITools/Profiling/atlasprofiling_16_0137.
                  html>`_ .
                  Note that in step 3 of configuring user permissions, the content in the msprof_data_collection.sh
                  script needs to be replaced with `msprof_data_collection.sh
                  <https://gitee.com/mindspore/mindspore/blob/master/docs/api/api_python/mindspore/script/
                  msprof_data_collection.sh>`_.

            output_path (str, optional): (Ascend only) Output data path. Default: ``"./dyn_profile_data"`` .
            poll_interval (int, optional): (Ascend only) The polling period of the monitoring process, in seconds.
                Default value: ``2``.

        Raises:
            RuntimeError: When create shared memory times exceeds max times.

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> import numpy as np
            >>> import mindspore as ms
            >>> from mindspore import nn
            >>> import mindspore.dataset as ds
            >>> from mindspore.profiler import DynamicProfilerMonitor
            >>>
            >>> class Net(nn.Cell):
            ...     def __init__(self):
            ...         super(Net, self).__init__()
            ...         self.fc = nn.Dense(2,2)
            ...     def construct(self, x):
            ...         return self.fc(x)
            >>>
            >>> def generator():
            ...     for i in range(2):
            ...         yield (np.ones([2, 2]).astype(np.float32), np.ones([2]).astype(np.int32))
            >>>
            >>> def train(net):
            ...     optimizer = nn.Momentum(net.trainable_params(), 1, 0.9)
            ...     loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
            ...     data = ds.GeneratorDataset(generator, ["data", "label"])
            ...     dynprof_cb = DynamicProfilerMonitor(cfg_path="./dyn_cfg", output_path="./dyn_prof_data")
            ...     model = ms.train.Model(net, loss, optimizer)
            ...     # register DynamicProfilerMonitor to model.train()
            ...     model.train(10, data, callbacks=[dynprof_cb])
        """

        def __init__(self, cfg_path=None, output_path="./dyn_profile_data", poll_interval=2, **kwargs):
            if DynamicProfilerUtils.is_dyno_mode() and cfg_path is not None:
                logger.warning("If you export 'MSMONITOR_USE_DAEMON=1', your 'cfg_path' parameter will be invalid!")
                cfg_path = None

            if not DynamicProfilerUtils.is_dyno_mode() and not isinstance(cfg_path, str):
                raise TypeError("If you set 'MSMONITOR_USE_DAEMON' to not 1, The cfg_path must be a string.")
            if not isinstance(output_path, str):
                logger.warning(f"The output_path must be a string, "
                               f"but got type {type(output_path)}, it will be set to './dyn_profile_data'.")
                output_path = "./dyn_profile_data"
            super().__init__(cfg_path, output_path, poll_interval, **kwargs)

        def _get_prof_args(self):
            """ Get prof_args py38"""
            byte_length = self._get_shm_byte_length()

            if byte_length == 0:
                return {}

            valid_bytes = self._shm.buf[:byte_length]
            return DynamicProfilerConfigContext.bytes_to_json(bytes(valid_bytes))

        def _get_shm_byte_length(self):
            byte_length = 0
            for i, byte in enumerate(self._shm.buf):
                if byte == 0:
                    byte_length = i
                    break
            return byte_length

        @no_exception_func()
        def _clean_resource(self):
            """Clean resource py38"""
            # stop profiler when stop_step over all train step
            if self._profiler:
                self._profiler.stop()
                ProfilerInterface.finalize()
                ProfilerInterface.clear()
                self._profiler = None
                logger.warning("Rank %d Dynamic profiler stop at end of training", self._rank_id)

            # join process
            if self._process:
                self._shared_loop_flag.value = False
                self._process.join()
                self._process = None
                logger.info("Rank %s process stop", self._rank_id)

            # clear shared memory
            if self._shm and self._is_create_process:
                try:
                    self._shm.close()
                    self._shm.unlink()
                    logger.info("Rank %s unlink shm", self._rank_id)
                except FileNotFoundError:
                    logger.warning("Rank %s unlink shm failed, may be removed", self._rank_id)
                self._shm = None

        @no_exception_func()
        def _finalize_dynolog(self):
            dyno_monitor_proxy = MsDynamicMonitorProxySingleton().get_proxy()
            dyno_monitor_proxy.finalize_dyno()
            logger.info("Rank %d finalize dynolog success !", self._rank_id)

        @no_exception_func()
        def _create_shm(self):
            """Create a json monitor process based on whether the SharedMemory is successfully created py38"""
            try_times = 10
            while try_times:
                try:
                    # Step 1: try to open shm file, first time shm not exists.
                    # Python incorrectly tracks shared memory even if it is not
                    # created by the process. The following patch is a workaround.
                    with patch("multiprocessing.resource_tracker.register",
                               lambda *args, **kwargs: None):
                        self._shm = shared_memory.SharedMemory(name=self._shm_name)
                    self._is_create_process = False
                    logger.info("Rank %d shared memory is connected.", self._rank_id)
                    break
                except FileNotFoundError:
                    try:
                        # Step 2: only one process can create shm successfully.
                        self._shm = shared_memory.SharedMemory(name=self._shm_name,
                                                               create=True, size=DynamicProfilerUtils.CFG_BUFFER_SIZE)
                        self._is_create_process = True
                        logger.info("Rank %d shared memory is created.", self._rank_id)
                        break
                    except FileExistsError:
                        # other process will go to step 1 and open shm file
                        try_times -= 1
                        logger.warning("Rank %d shared memory create failed, "
                                       "retry times = %d.", self._rank_id, try_times)
                        time.sleep(random.uniform(0, 0.02))  # sleep 0 ~ 20 ms
                except Exception as e:  # pylint: disable=W0703
                    # shm open failed because of other process create shm not finished
                    try_times -= 1
                    logger.warning("Rank %d shared memory open failed, error: %s, retry times = %d",
                                   self._rank_id, str(e), try_times)
                    time.sleep(random.uniform(0, 0.02))  # sleep 0 ~ 20 ms

            if try_times <= 0:
                raise RuntimeError(f"Rank {self._rank_id} failed to create shared memory.")

else:
    import mmap


    class DynamicProfilerMonitor(DynamicProfilerMonitorBase):
        r"""
        This class to enable the dynamic profiler monitoring of MindSpore neural networks.

        Args:
            cfg_path (str): Dynamic profiler json config file directory. The requirement is a shared path
                that can be accessed by all nodes.
            output_path (str, optional): Output data path. Default: ``"./dyn_profile_data"`` .
            poll_interval (int, optional): The polling period of the monitoring process, in seconds.
                Default value: ``2``.

        Raises:
            RuntimeError: When create shared memory times exceeds max times.

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> import numpy as np
            >>> import mindspore as ms
            >>> from mindspore import nn
            >>> import mindspore.dataset as ds
            >>> from mindspore.profiler import DynamicProfilerMonitor
            >>>
            >>> class Net(nn.Cell):
            ...     def __init__(self):
            ...         super(Net, self).__init__()
            ...         self.fc = nn.Dense(2,2)
            ...     def construct(self, x):
            ...         return self.fc(x)
            >>>
            >>> def generator():
            ...     for i in range(2):
            ...         yield (np.ones([2, 2]).astype(np.float32), np.ones([2]).astype(np.int32))
            >>>
            >>> def train(net):
            ...     optimizer = nn.Momentum(net.trainable_params(), 1, 0.9)
            ...     loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
            ...     data = ds.GeneratorDataset(generator, ["data", "label"])
            ...     dynprof_cb = DynamicProfilerMonitor(cfg_path="./dyn_cfg", output_path="./dyn_prof_data")
            ...     model = ms.train.Model(net, loss, optimizer)
            ...     # register DynamicProfilerMonitor to model.train()
            ...     model.train(10, data, callbacks=[dynprof_cb])
        """

        def __init__(self, cfg_path=None, output_path="./dyn_profile_data", poll_interval=2, **kwargs):
            if DynamicProfilerUtils.is_dyno_mode() and cfg_path is not None:
                logger.warning("If you export 'MSMONITOR_USE_DAEMON=1', your 'cfg_path' parameter will be invalid!")
                cfg_path = None

            if not DynamicProfilerUtils.is_dyno_mode() and not isinstance(cfg_path, str):
                raise TypeError("If you set 'MSMONITOR_USE_DAEMON' to not 1, The cfg_path must be a string.")

            if not isinstance(output_path, str):
                logger.warning(f"The output_path must be a string, "
                               f"but got type {type(output_path)}, it will be set to './dyn_profile_data'.")
                output_path = "./dyn_profile_data"
            self._cfg_path = cfg_path
            self._shm_name = time.strftime("DynamicProfileShm%Y%m%d%H", time.localtime())
            self._shm_dir = (
                "/dev/shm" if DynamicProfilerUtils.is_dyno_mode()
                else os.path.join(self._cfg_path, "shm")
            )
            PathManager.make_dir_safety(self._shm_dir)
            self._shm_path = os.path.realpath(os.path.join(self._shm_dir, self._shm_name))

            super().__init__(cfg_path, output_path, poll_interval, **kwargs)
            logger.warning("Dynamic profiler is not work well on python 3.7x, "
                           "please update to python 3.8+ for better performance.")

        def _get_prof_args(self):
            """ Get prof_args py37"""
            self._shm.seek(0)
            return DynamicProfilerConfigContext.bytes_to_json(
                bytes(self._shm.read(DynamicProfilerUtils.CFG_BUFFER_SIZE)))

        @no_exception_func()
        def _clean_resource(self):
            """Clean resource py37"""
            # stop profiler when stop_step over all train step
            if self._profiler:
                self._profiler.stop()
                ProfilerInterface.finalize()
                ProfilerInterface.clear()
                self._profiler = None
                logger.warning("Rank %d Dynamic profiler stop at end of training", self._rank_id)

            # join process
            if self._process:
                self._shared_loop_flag.value = False
                self._process.join()
                self._process = None
                logger.info("Rank %s process stop", self._rank_id)

            # clear shared memory
            if self._shm and self._is_create_process:
                try:
                    self._shm.close()
                    if self._memory_mapped_file and not self._memory_mapped_file.closed:
                        self._memory_mapped_file.close()
                    elif self.fd:
                        os.close(self.fd)
                    PathManager.remove_file_safety(self._shm_path)
                    logger.info("Rank %s unlink shm", self._rank_id)
                except FileNotFoundError:
                    logger.warning("Rank %s unlink shm failed, may be removed", self._rank_id)
                self._shm = None

        @no_exception_func()
        def _create_shm(self):
            """Create a json monitor process based on whether the SharedMemory is successfully created py37"""

            try_times = 10
            while try_times:
                try:
                    # Step 1: try to open fd, first time fd not exists.
                    self.fd = os.open(self._shm_path, os.O_EXCL | os.O_RDWR,
                                      stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP)
                    self._memory_mapped_file = os.fdopen(self.fd, 'rb')
                    self._shm = mmap.mmap(self._memory_mapped_file.fileno(),
                                          length=DynamicProfilerUtils.CFG_BUFFER_SIZE)
                    self._is_create_process = False
                    logger.info("Rank %d shared memory is connected.", self._rank_id)
                    break
                except ValueError:
                    time.sleep(0.02)
                except FileNotFoundError:
                    try:
                        # Step 2: only one process can create fd successfully.
                        fd = os.open(self._shm_path, os.O_CREAT | os.O_EXCL | os.O_RDWR,
                                     stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP)

                        # Init mmap file need to write data
                        with os.fdopen(fd, 'wb') as f:
                            data_instance = DynamicProfilerConfigContext({})
                            byte_data = data_instance.to_bytes()
                            f.write(byte_data)

                        # create mmap
                        self.fd = os.open(self._shm_path, os.O_EXCL | os.O_RDWR,
                                          stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP)
                        self._memory_mapped_file = os.fdopen(self.fd, 'rb')
                        self._shm = mmap.mmap(self._memory_mapped_file.fileno(), length=DynamicProfilerUtils.
                                              CFG_BUFFER_SIZE)
                        self._is_create_process = True
                        logger.info("Rank %d shared memory is created.", self._rank_id)
                        break
                    except FileExistsError:
                        # other process will go to step 1 and open shm file
                        try_times -= 1
                        logger.warning("Rank %d shared memory create failed, "
                                       "retry times = %d.", self._rank_id, try_times)
                        time.sleep(random.uniform(0, 0.02))  # sleep 0 ~ 20 ms

            if try_times <= 0:
                raise RuntimeError("Failed to create shared memory.")

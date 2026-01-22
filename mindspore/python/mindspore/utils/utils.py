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
"""mindspore utils."""
from __future__ import absolute_import

import os
import json
from mindspore import log as logger
from mindspore import context
from mindspore import _checkparam as Validator
from mindspore.common import dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.graph.api import jit_class
from mindspore._c_expression import _tft_start_record_threads, _tft_finish_record_threads
from mindspore._c_expression import set_is_reboot_node, tft_register_config, set_reboot_type


@jit_class
class ExitByRequest:
    """
    Gracefully exits the training process after get exit request.
    """

    def __init__(self):
        super().__init__()
        from mindspore.communication.management import get_group_size
        self.all_reduce = P.AllReduce()
        self.equal = P.Equal()
        self.assign = P.Assign()
        self.reduce_all = P.ReduceAll(keep_dims=False)
        self.group_size = get_group_size()
        self.is_distributed = self.group_size > 1
        if self.is_distributed:
            self.base = Tensor([self.group_size], dtype=mstype.int32)
        self.base1 = Tensor([1], mstype.int32)
        self.true = Tensor(True, mstype.bool_)

    def exit_by_request(self, grad, init_value, exit_value):
        """
        update GracefulExit flag by Assign op, the value is the output of AllReduce op
        :param grad: grad of net, or output of opt
        :param init_value: input value of AllReduce, a parameter
        :param exit_value: graceful exit value(out of AllReduce), update by Assign op
        :return: grad
        """
        if self.is_distributed:
            all_status = self.all_reduce(init_value)
            equal = self.equal(all_status, self.base)
            reduce_all = self.reduce_all(equal)
            grad = F.depend(grad, self.assign(exit_value, reduce_all))
        return grad


class TFTCommValue:
    """Config values"""
    ENABLE_MINDX = ['TTP:1', 'UCE:1', 'ARF:1', 'TSP:1', 'HCCE:1', 'RSC:1']  # support mindx to schedule
    NEED_MINDIO = ["TTP:1", "UCE:1", "ARF:1", "TSP:1", "HCCE:1"]  # need mindio-ttp pkg
    DISABLE_WATCHDOG = ['ARF:1', 'TSP:1', 'HCCE:1']  # close watchdog


def _getenv():
    """Get env """
    tft_env = os.getenv("MS_ENABLE_TFT", "").strip()
    thm_env = os.getenv("MS_ENABLE_THM", "").strip()
    return tft_env, thm_env


def _parser_tft_and_thm_env():
    """Parser all config: tft, thm ..."""
    tft_env, thm_env = _getenv()
    tft_envs = tft_env.replace("{", "").replace("}", "").strip().split(",")
    thm_envs = thm_env.replace("{", "").replace("}", "").strip().split(",")
    all_config = {}
    for item in tft_envs:
        if item == "":
            continue
        key_v = item.split(":")
        all_config[key_v[0].strip()] = key_v[1].strip()

    for item in thm_envs:
        if item == "":
            continue
        key_v = item.split(":")
        if key_v[0] == "HCCL_STATUS_SAVE_CONFIG":
            with open(key_v[1].strip("'\""), 'r', encoding='utf-8') as j:
                json_values = json.load(j)
                for key, val in json_values.items():
                    if key == "HCCL_STATUS_SAVE_PATH" and not os.path.isabs(str(val)):
                        logger.warning(
                            f"HCCL_STATUS_SAVE_PATH should be absolute path, but get: {val}, Using default path:'/tmp'")
                        val = "/tmp"
                    key = "CCAE_" + key
                    all_config[key] = val
            continue
        all_config[key_v[0].strip()] = key_v[1].strip()
    if all_config.get("ARF") == "1":
        logger.warning("Disable hccl_watchdog and turn on TTP when using ARF.")
        all_config["HCCL_WATCHDOG"] = "0"
        all_config["TTP"] = "1"
    if all_config.get("HCCL_STATUS_SAVE") == "1":
        os.environ["HCCL_STATUS_SAVE"] = "1"
        os.environ["HCCL_STATUS_SAVE_PATH"] = all_config.get("CCAE_HCCL_STATUS_SAVE_PATH")
        os.environ["HCCL_STATUS_SAVE_INTERVAL"] = str(all_config.get("CCAE_HCCL_STATUS_SAVE_INTERVAL"))
    tft_register_config(all_config)


class RSCPluginHandle:
    """Third party controller handler"""

    def __init__(self):
        self.enable = False
        self.tft_env, _ = _getenv()
        self._check_env()
        self.msmgr = None
        self.init_taskd_agent = None
        self.start_taskd_agent = None
        self.register_func = None
        self.using_agent = False

    def _check_env(self):
        """Check env"""
        self.enable = any(v in self.tft_env for v in TFTCommValue.ENABLE_MINDX)

    def check_enable(self):
        """Check env"""
        return self.enable

    def _register_by_agent(self, func_map):
        """ register by taskd agent"""
        try:
            from taskd.api.taskd_agent_api import init_taskd_agent, start_taskd_agent, register_func
            self.init_taskd_agent = init_taskd_agent
            self.start_taskd_agent = start_taskd_agent
            self.register_func = register_func
        except ImportError as e:
            logger.warning(f"Import task agent: {str(e)}, try to using mindx plugin.")
            return False
        try:
            logger.warning("register callbacks to taskd agent")
            if not self.init_taskd_agent({"Framework": "MindSpore"}):
                logger.warning("Init taskd agent failed, try to using mindx plugin.")
                return False
            for name, func in func_map.items():
                self.register_func(name, func)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(f"Register callback func failed: {str(e)}, try to using mindx plugin.")
            return False
        self.using_agent = True
        return True

    def _register_by_plugin(self, func_map):
        """ register by mindx msrun_plugin"""
        # will delete in the future
        self.using_agent = False
        try:
            from taskd.python.framework.agent.ms_mgr.msrun_plugin import MSRunPlugin
            self.msmgr = MSRunPlugin()
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(f"Import mindx failed: {str(e)}, process controlled by msrun.")
            return False
        try:
            for name, func in func_map.items():
                self.msmgr.register_callbacks(name, func)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(f"Register callback func failed: {str(e)}, process controlled by msrun")
            return False
        return True

    def register_callback(self, func_map: dict):
        """Register function"""
        if not isinstance(func_map, dict):
            raise ValueError(f"The value of 'func_map' should be a dict, bug got:{func_map}.")
        if self._register_by_agent(func_map):
            return True
        if self._register_by_plugin(func_map):
            return True
        return False

    def start(self):
        """Start execute taskd agent"""
        if self.using_agent:
            logger.warning("start by taskd agent")
            self.start_taskd_agent()
        else:
            logger.warning("start by mindx")
            if self.msmgr is None:
                raise RuntimeError("Mindx unavailable, can not start training.")
            self.msmgr.start()


class TftHandle:
    """TftHandle class"""

    def __init__(self):
        _parser_tft_and_thm_env()
        self._controller_ip = None
        self._controller_rank_id = None
        self._controller_port = None
        self.tft = None
        self.enable_mindx = False
        self.tft_notify_controller_prepare_action = None
        self.tft_notify_controller_change_strategy = None

    def get_tft(self):
        """return tft handle"""
        return self.tft

    def unregister_tft(self):
        """unregister tft"""
        cur_rank = int(os.getenv("MS_NODE_ID"))  # from msrun
        if cur_rank == self._controller_rank_id and not self.enable_mindx:
            self.tft.tft_destroy_controller()
        self.tft.tft_destroy_processor()

    def _mindx_stub(self):
        """stub func for mindx"""
        from mindio_ttp.controller_ttp import (tft_register_mindx_callback,
                                               tft_notify_controller_stop_train,
                                               tft_notify_controller_prepare_action,
                                               tft_notify_controller_on_global_rank,
                                               tft_notify_controller_change_strategy)

        self.tft_notify_controller_prepare_action = tft_notify_controller_prepare_action
        self.tft_notify_controller_change_strategy = tft_notify_controller_change_strategy

        def report_fault_ranks_func(error_rank_dict):
            tft_notify_controller_stop_train(error_rank_dict)
            return 0

        def report_stop_complete_func(code, msg, error_rank_dict):
            tft_notify_controller_on_global_rank(error_rank_dict)
            return 0

        def report_strategies_func(error_rank_dict, strategy_list):
            tft_notify_controller_change_strategy(strategy_list[-1])
            return 0

        def report_result(code, msg, error_rank_dict, curr_strategy):
            if code != 0:
                tft_notify_controller_change_strategy('dump')
            return 0

        logger.warning('Stub for mindx.')
        tft_register_mindx_callback('report_fault_ranks', report_fault_ranks_func)
        tft_register_mindx_callback('report_stop_complete', report_stop_complete_func)
        tft_register_mindx_callback('report_strategies', report_strategies_func)
        tft_register_mindx_callback('report_result', report_result)
        logger.warning('Stub register mindx func success.')

    def init(self, **kwargs):
        """
        TFT handle init fun. Mainly used to initialize the mindio component.

        Args:
            **kwargs: Reserved parameters.
        """
        tft_env, _ = _getenv()
        tft_enabled = any([opt in tft_env for opt in TFTCommValue.NEED_MINDIO])  # pylint: disable=R1729
        if not tft_enabled:
            raise ValueError(F"MindIO TFT register need custom switch on one of:{TFTCommValue.NEED_MINDIO}")
        if "ARF:1" in tft_env:
            if "TTP:1" not in tft_env:
                logger.warning("Turn on TTP config when using ARF.")
                tft_env = tft_env.replace("{", "").replace("}", "")
                all_opts = [part.strip() for part in tft_env.split(",")] + ["TTP:1"]
                os.environ["MS_ENABLE_TFT"] = "{" + ",".join(all_opts) + "}"
            os.environ["MS_ENABLE_RECOVERY"] = "1"

        device_target = context.get_context("device_target")
        if device_target != "Ascend":
            logger.warning(f"MindIO adataper only support on Ascend device but got device {device_target}!")
            return

        ctrl_port = int(os.getenv("MS_TFT_PORT"))
        ctrl_ip = os.getenv("MS_TFT_IP", "")
        Validator.check_non_negative_int(ctrl_port)
        self._controller_ip = ctrl_ip
        self._controller_rank_id = 0
        self._controller_port = ctrl_port
        try:
            from mindio_ttp import framework_ttp as tft
            self.tft = tft
        except BaseException as e:
            raise ModuleNotFoundError(f"Module not found. Detail info {str(e)}")  # pylint: disable=W0707
        world_size = int(os.getenv("MS_WORKER_NUM"))  # from msrun
        cur_rank = int(os.getenv("MS_NODE_ID"))  # from msrun
        enable_local_copy = False
        enable_arf = True if "ARF:1" in tft_env else False  # pylint: disable=simplifiable-if-expression
        enable_tls = False
        tls_key_dir = ""
        self.enable_mindx = os.getenv("MINDX_TASK_ID")
        # enable mindx, no need create controller
        if cur_rank == self._controller_rank_id and self.enable_mindx is None:
            logger.info(f"Begin to start tft controller on rank_id:{cur_rank}")
            if enable_arf:
                self._mindx_stub()
            self.tft.tft_init_controller(cur_rank, world_size, enable_local_copy, enable_arf=enable_arf)
            self.tft.tft_start_controller(self._controller_ip, self._controller_port, enable_tls, tls_key_dir)
            logger.info("Finish start tft controller.")

        logger.info("Begin to start tft processor.")
        _tft_start_record_threads()
        self.tft.tft_init_processor(cur_rank, world_size, enable_local_copy, enable_tls, tls_key_dir,
                                    enable_arf=enable_arf)
        self.tft.tft_start_processor(self._controller_ip, self._controller_port)
        _tft_finish_record_threads()
        logger.info("Finished start tft processor.")
        if self.tft.tft_is_reboot_node():
            logger.warning("tft report reboot init finish ")
            tft.tft_report_error(tft.ReportState.RS_INIT_FINISH.value)
            set_is_reboot_node(True)
            try:
                reboot_type = self.tft.tft_get_reboot_type()
                logger.warning(f"get reboot type:{reboot_type}")
            except Exception as e:  # pylint: disable=broad-except
                logger.warning(f"get reboot type failed:{e}, only support arf")
                reboot_type = "arf"
            if reboot_type == "arf":
                set_reboot_type("arf")
                ret = tft.tft_wait_next_action()
                if ret != tft.Action.RETRY.value:
                    raise RuntimeError("ARF init failed!")  # pylint: disable=W0707
            else:
                set_reboot_type("hot_switch")
            logger.warning("tft reboot success.")


_tft_handler = TftHandle()

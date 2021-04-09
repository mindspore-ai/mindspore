# Copyright 2021 Huawei Technologies Co., Ltd
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
"""tuner process"""
import os
import datetime
import json
import sys
import traceback
from te.platform.cce_conf import te_set_version
from te.platform.fusion_manager import set_current_op_name
from te.platform.fusion_util import fusion_op, dump_fusion_json
from te.platform.parallel_compilation import init_multi_process_env, get_finished_compilation_task, \
    deinit_multi_process_env, dispatch_autotune_task, start_ga_multi_process, import_py_module
import auto_tune
from schedule_search.rl_online_tune import rl_tune_init, dispatch_fusion_tune_task, dispatch_single_tune_task, \
    rl_tune_deinit
from mindspore import log
from .compiler import build_op
from .re_construct_json import single_to_fusion, fusion_to_fusion

TE_LOG_LEVEL = ["DEBUG", "INFO", "WARNING", "ERROR"]
RL_COMPILE = "RL_COMPILE"
RL_OFFLINE = "RL_OFFLINE"
RL_ONLINE = "RL_ONLINE"
OP_BUILD = "compile"

PLATFORM_FLAG = ["ascend310", "ascend910", "Hi3796CV300ES", "ascend710", "ascend610", "Hi3796CV300CS", "SD3403"]


class TbeTuner:
    """tbe tuner for ga tune or rl tune"""

    def __init__(self, offline_tune, tune_mode):
        self.offline_tune = offline_tune
        self.tune_init = False
        self.rl_init = False
        self.multi_init = False
        self.offline_dump_path = "./tune_dump"
        if os.environ.get("TUNE_DUMP_PATH") is not None:
            self.offline_dump_path = os.getenv("TUNE_DUMP_PATH", "")
        self._creating_custom_path(tune_mode)
        self.fusion_need_sync = 0
        self.module_list = {}

    def init_tune_interface(self, json_str, process_num):
        """
        Initialize tuner interface
        :param json_str: ori json
        :param process_num : process num for tuner
        :return: bool True or False
        """
        json_info = json.loads(json_str)
        soc_info = self.get_soc_info(json_info)
        cur_cce_product_params = te_set_version(*soc_info)
        if cur_cce_product_params is None:
            log.warning("Set Soc Info failed.")
        tune_mode = self.get_tune_mode(json_info)
        ret = self.parallel_compilation_init(soc_info, tune_mode, process_num)
        if not ret:
            log.error("Init parallel compilation env failed")
            return False

        return True

    def deinit(self):
        """
        DeInitialize tuner interface
        """
        if self.multi_init:
            deinit_multi_process_env()
        if self.rl_init:
            rl_tune_deinit()

    def get_tune_mode(self, json_info):
        """
        Get the corresponding tune mode from op json and env info
        :param json_info: ori json
        :return: NO_TUNE RL_TUNE GA_TUNE or RL,GA
        """
        tune_mode = json_info["SocInfo"]["autoTilingMode"]
        if self.offline_tune:
            tune_mode = "RL"
        return tune_mode

    def __directory_creation(self, path, concat_path):
        """
        Create directory
        """
        path = os.path.join(path, concat_path)
        if not os.path.isdir(path):
            os.makedirs(path, 0o750)
        return path

    def __creating_default_custom_path(self, tune_mode, base_custom_path):
        """
        Create default custom path
        """
        base_custom_path = self.__directory_creation(base_custom_path, "data")
        tune_flag = []
        if "RL" in tune_mode:
            tune_flag.append("rl")
        if "GA" in tune_mode:
            tune_flag.append("tiling")

        for tune_path in tune_flag:
            real_path = self.__directory_creation(base_custom_path, tune_path)
            for soc_version in PLATFORM_FLAG:
                final_path = self.__directory_creation(real_path, soc_version)
                final_path = self.__directory_creation(final_path, "custom")

    def _creating_custom_path(self, tune_mode):
        """
        Create custom path
        """
        if "NO_TUNE" in tune_mode:
            return

        base_custom_path = os.getenv("TUNE_BANK_PATH", None)
        tune_bank_flag = True
        if not base_custom_path:
            base_custom_path = os.path.dirname(os.path.realpath(auto_tune.__file__))
            base_custom_path = os.path.realpath(os.path.join(base_custom_path, "../../../"))
            tune_bank_flag = False

        if not os.path.isdir(base_custom_path):
            log.error("Check whether the tuning path [{}] exists.".format(base_custom_path))
            return
        if not os.access(base_custom_path, os.R_OK | os.W_OK | os.X_OK):
            log.error("Check whether the permission on the tuning path [{}] is correct.".format(base_custom_path))
            return

        if not tune_bank_flag:
            self.__creating_default_custom_path(tune_mode, base_custom_path)

    def get_soc_info(self, json_info):
        """
        Get soc info
        :param json_info: ori json
        :return: soc info
        """
        soc_param = {}
        soc_param["op_impl_mode"] = json_info["SocInfo"]["op_impl_mode"]
        soc_param["op_debug_level"] = json_info["SocInfo"]["op_debug_level"]
        soc_param["op_impl_mode_list"] = json_info["SocInfo"]["op_impl_mode_list"]
        soc_param["op_debug_dir"] = ''
        soc_param["vector_fp_ceiling"] = ''
        soc_param['mdl_bank_path'] = ''
        soc_param['op_bank_path'] = ''

        soc_info = []
        soc_info.append(json_info["SocInfo"]["socVersion"])
        soc_info.append(json_info["SocInfo"]["coreType"])
        soc_info.append(json_info["SocInfo"]["coreNum"])
        soc_info.append(json_info["SocInfo"]["l1Fusion"])
        soc_info.append(json_info["SocInfo"]["l2Mode"])
        soc_info.append(json_info["SocInfo"]["l2Fusion"])
        soc_info.append(soc_param)

        return soc_info

    def check_te_log(self, te_log_level):
        """
        Check te log level
        :param te_log_level:
        :return:
        """
        res = True
        if te_log_level.isdigit() and int(te_log_level) >= len(TE_LOG_LEVEL):
            log.error(f"Invalid environment TE_LOGLEVEL, the value should be in [0, 4) if it is a digit, but got : "
                      f"{te_log_level}")
            res = False
        elif te_log_level.upper() not in TE_LOG_LEVEL:
            log.error(f"Invalid environment TE_LOGLEVEL, the value should be one of [DEBUG, INFO, WARNING, ERROR] "
                      f"if it is a string, but got :{te_log_level}")
            res = False
        return res

    def parallel_compilation_init(self, soc_info, tune_mode, process_num):
        """
        Initialize parallel compilation framework for tuner
        :param soc_info: soc info
        :param tune_mode: tuner mode
        :param process_num : process num for tuner
        :return: bool True or False
        """
        env_count = process_num
        if "TE_PARALLEL_COMPILER" in os.environ:
            env_count = os.getenv("TE_PARALLEL_COMPILER")
            log.info("TE_PARALLEL_COMPILER is set to {}".format(env_count))
            if int(env_count) > process_num:
                env_count = process_num
                log.info("change process count to {}".format(process_num))
        os.environ["TE_PARALLEL_COMPILER"] = str(int(env_count))
        pid_str = os.getpid()
        time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f')[:-3]
        pid_ts = "{}_pid{}".format(time_str, pid_str)

        embedding = False
        enable_event = False
        te_log_level = os.environ.get("TE_LOGLEVEL")
        glog_level = os.environ.get("GLOG_v")
        if glog_level is not None and te_log_level is None:
            os.environ["TE_LOGLEVEL"] = TE_LOG_LEVEL[int(glog_level)]
            global_loglevel = int(glog_level)
        elif glog_level is None and te_log_level is None:
            os.environ["TE_LOGLEVEL"] = TE_LOG_LEVEL[2]
            global_loglevel = 3
        else:
            if not self.check_te_log(te_log_level):
                return False
            global_loglevel = int(te_log_level) if te_log_level.isdigit() else TE_LOG_LEVEL.index(te_log_level.upper())
        ret = init_multi_process_env(embedding, soc_info, tune_mode, global_loglevel, enable_event, pid_ts)
        if ret is None:
            log.error("Init multiprocess env failed")
            return False
        self.multi_init = True
        process_count = ret[0]
        log.info("Init multiprocess env success with {} process".format(process_count))
        if "RL" in tune_mode:
            res_queue = ret[1]
            live_checker = ret[2]
            termin_event = ret[3]
            ret = rl_tune_init(soc_info, res_queue, live_checker, termin_event, global_loglevel, pid_ts)
            if not ret:
                log.error("RL env init failed!")
                return False
            self.rl_init = True
            log.info("RL Tune init success.")
        if "GA" in tune_mode:
            start_ga_multi_process(tune_mode)
            log.info("GA Tune init success.")
        return True

    def sync_fusion_env(self):
        """
        Sync fusion env
        :return: None
        """
        if self.fusion_need_sync == 0:
            return

        module_using = []
        for key, value in self.module_list.items():
            if value > 0:
                module_using.append(str(key))
            self.module_list[key] = 0

        module_str = ",".join(module_using)
        import_py_module(module_str)
        self.fusion_need_sync = 0

    def rl_tune(self, task_id, op_json):
        """
        RL tune for single op and fusion op
        :param task_id: task id for this op to tune
        :param op_json: op's info
        :return: tune result
        """
        json_info = json.loads(op_json)
        if "fusion_op" in json_info:
            self.sync_fusion_env()
            ret = self.fusion_rl_tune(task_id, json_info)
        else:
            ret = self.single_rl_tune(task_id, json_info)
        return ret

    def ga_tune(self, task_id, op_json):
        """
        GA tune for single op and fusion op
        :param task_id: task id for this op to tune
        :param op_json: op's info
        """
        json_info = json.loads(op_json)
        if "fusion_op" in json_info:
            self.sync_fusion_env()
            self.fusion_ga_tune(task_id, json_info)
        else:
            self.single_ga_tune(task_id, json_info)

    def single_rl_tune(self, task_id, json_info):
        """
        RL tune for single op
        :param task_id: task id for this op to tune
        :param json_info: op's info
        :return: tune result
        """
        if self.offline_tune:
            converted_json = single_to_fusion(json.dumps(json_info), tune_mode="RL")
        op_type = json_info['op_info']['name']
        kernel_name = json_info['op_info']['kernel_name']
        full_name = json_info['op_info']['full_name']
        tune_mode = "RL"
        set_current_op_name(kernel_name)
        # todo build with build_single_op_from_c
        base_kernel = './kernel_meta/' + kernel_name + '.o'
        job_type = RL_COMPILE
        compile_info = None
        try:
            compile_info, op_args, op_module_name = build_op(OP_BUILD, json.dumps(json_info), tune_mode)
        # pylint: disable=broad-except
        except Exception:
            exc_type, exc_value, _ = sys.exc_info()
            log.error(
                "exc_type:{}, exc_value:{}, exc_traceback:{}".format(exc_type, exc_value, traceback.format_exc()))
            return False, job_type, compile_info
        if self.offline_tune:
            job_type = RL_OFFLINE
            dump_fusion_json(converted_json, self.offline_dump_path)
        else:
            job_type = RL_ONLINE
        graph_id = 0
        l1size = 0  # todo need to verify
        ret = dispatch_single_tune_task(graph_id, task_id, l1size, base_kernel, kernel_name, full_name,
                                        op_module_name + "@" + op_module_name, op_type, op_type, op_args)

        self.module_list[op_module_name] = 1
        self.fusion_need_sync += 1
        return ret, job_type, compile_info

    def fusion_rl_tune(self, task_id, json_info):
        """
        RL tune for fusion op
        :param task_id: task id for this op to tune
        :param json_info: op's info
        :return: tune result
        """
        if 'fusion_op' not in json_info or not json_info['fusion_op']:
            raise ValueError("Json string Errors, key:fusion_op not found.")
        kernel_name = json_info["fusion_op"]["fusion_op_name"]
        full_name = json_info["fusion_op"]["full_name"]
        set_current_op_name(kernel_name)
        converted_json = fusion_to_fusion(json.dumps(json_info), tune_mode="RL")
        job_type = RL_COMPILE
        base_kernel = './kernel_meta/' + kernel_name + '.o'
        compile_info = None
        try:
            fusion_op(converted_json)
        # pylint: disable=broad-except
        except Exception:
            exc_type, exc_value, _ = sys.exc_info()
            log.error(
                "exc_type:{}, exc_value:{}, exc_traceback:{}".format(exc_type, exc_value, traceback.format_exc()))
            return False, job_type, compile_info
        if self.offline_tune:
            job_type = RL_OFFLINE
            dump_fusion_json(converted_json, self.offline_dump_path)
        else:
            job_type = RL_ONLINE
        graph_id = 0
        l1size = 0
        ret = dispatch_fusion_tune_task(graph_id, task_id, l1size, base_kernel, kernel_name, full_name,
                                        converted_json)
        return ret, job_type, compile_info

    def fusion_ga_tune(self, task_id, json_info):
        """
        GA tune for fusion op
        :param task_id: task id for this op to tune
        :param json_info: op's info
        """
        if 'fusion_op' not in json_info or not json_info['fusion_op']:
            raise ValueError("Json string Errors, key:fusion_op not found.")
        kernel_name = json_info["fusion_op"]["fusion_op_name"]
        converted_json = fusion_to_fusion(json.dumps(json_info), tune_mode="GA")
        graph_id = 0
        l1size = 0
        dispatch_autotune_task(graph_id, task_id, l1size, converted_json, [], kernel_name)

    def single_ga_tune(self, task_id, json_info):
        """
        GA tune for single op
        :param task_id: task id for this op to tune
        :param json_info: op's info
        """
        converted_json = single_to_fusion(json.dumps(json_info), tune_mode="GA")
        graph_id = 0
        l1size = 0
        kernel_name = json.loads(converted_json)["fusion_op_name"]
        dispatch_autotune_task(graph_id, task_id, l1size, converted_json, [], kernel_name)

    def get_finish_tasks(self):
        """
        Get finish task from parallel compilation framework
        :return task info list
        """
        ret = get_finished_compilation_task(0)
        return ret

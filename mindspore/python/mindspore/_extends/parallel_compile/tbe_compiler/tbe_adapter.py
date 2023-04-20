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
"""tbe adapter to adapt te/topi/auto-tune python api """
from __future__ import absolute_import
import json
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone

from tbe.common.rl_bank.bank_manager import set_current_op_name
from tbe.common.repository_manager.interface import cann_kb_finalize, cann_kb_init
from te.platform.cce_conf import te_set_version
from te.platform.cce_policy import set_L1_info
from te_fusion.compile_task_manager import dispatch_prebuild_task, dispatch_single_op_compile_task, \
    dispatch_fusion_op_compile_task, dispatch_autotune_task, sync_op_tune_params
from te_fusion.compile_task_manager import sync_syspath
from te_fusion.fusion_manager import call_op_func, clear_fusion_params, check_op_impl_mode, \
    save_op_params, build_single_op_from_c, op_params_to_json
from te_fusion.fusion_util import dump_fusion_json
from te_fusion.parallel_compilation import init_multi_process_env, start_ga_multi_process, deinit_multi_process_env, \
    get_finished_compilation_task

from .tbe_helper import get_soc_info, assemble_op_args, get_compute_op_list, get_options_info, get_fuzz_build_info, \
    adjust_custom_op_info, pack_op_args, get_module_name, get_real_op_debug_level, LocalLock
from .tbe_job import TbeJob, JobStatus


def _tune_init(job: TbeJob):
    """
    Tune Initialize
    :param job:
    :return:
    """
    auto_tiling_mode = job.content["SocInfo"]["autoTilingMode"]
    offline_tune = job.content["SocInfo"]["offlineTune"]
    op_bank_update = job.content["SocInfo"]["op_bank_update"]
    tune_dump_path = job.content["TuneInfo"]["tune_dump_path"]
    tune_bank_path = job.content["TuneInfo"]["tune_bank_path"]
    need_ga = bool("GA" in auto_tiling_mode)
    need_rl = bool("RL" in auto_tiling_mode)
    if offline_tune:
        os.environ["ENABLE_TUNE_DUMP"] = "TRUE"
    if op_bank_update:
        sync_op_tune_params("tbe.common.tiling.tiling_api", "reset_repository", False, "")

    if need_ga or need_rl or offline_tune:
        res = __init_tune_env(job, need_ga)
        if not res:
            return False
    else:
        return True

    if tune_dump_path:
        os.environ["TUNE_DUMP_PATH"] = str(tune_dump_path)
    if tune_bank_path:
        os.environ["TUNE_BANK_PATH"] = str(tune_bank_path)
    res = _creating_custom_path(job)
    return res


def _cann_kb_init(job: TbeJob):
    """
    database init
    :param job:
    :return:
    """
    sys_config = {"soc_version": job.soc_version, "core_num": job.core_num}
    load_config = {"op_bank_path": job.op_bank_path}
    kb_type = None
    res = cann_kb_init(sys_config, load_config, kb_type)
    return res


def _cann_kb_finalize():
    """
    database finalize
    :return:
    """
    res = cann_kb_finalize()
    return res


def _remove_cache(job: TbeJob):
    """
    :param job: remove cache file:[*.json, *.o, *.info, *.cce] when "op_debug_level" is "0"
                op_debug_level: representation the env MS_COMPILER_OP_LEVEL
    :return:
    """
    op_debug_level = job.content["SocInfo"]["op_debug_level"]
    op_debug_dir = job.content["SocInfo"]["op_debug_dir"]
    if op_debug_level != "0":
        return
    root_path = os.path.abspath(op_debug_dir)
    if os.path.exists(root_path):
        real_path = os.path.join(root_path, "kernel_meta/")
        shutil.rmtree(real_path)


def __directory_creation(path, concat_path):
    """
    Create directory
    """
    path = os.path.join(path, concat_path)
    if not os.path.isdir(path):
        os.makedirs(path, 0o750)
    return path


def __init_tune_env(job, need_ga):
    """
    Initialize tune env
    """
    try:
        import auto_tune.auto_tune_main as at_atm
        from schedule_search.rl_online_tune import rl_tune_init  # pylint: disable=unused-import
    except ImportError:
        msg = "TBEException", \
              "No module named `auto_tune` or `schedule_search`. If you want tune your op's performance," \
              "please configure `auto_tune` or `schedule_search` related environment variables." \
              "Try to set the following environment variables:" \
              "export fwk_path=/usr/local/Ascend/latest" \
              "export PYTHONPATH=${fwk_path}/python/site-packages:$PYTHONPATH" \
              "export PYTHONPATH=${fwk_path}/python/site-packages/auto_tune.egg/auto_tune:$PYTHONPATH" \
              "export PYTHONPATH=${fwk_path}/python/site-packages/schedule_search.egg:$PYTHONPATH"
        job.error(msg)
        return False
    finally:
        pass

    if need_ga:
        res = at_atm.ga_tune_init()
        if not res:
            job.error("check soc version failed in tune init")
            job.error("GATune run Failed. Run .o Failed, because soc_version doesn't match the device")
            return False
    return True


def __creating_default_custom_path(auto_tiling_mode, base_custom_path):
    """
    Create default custom path
    """
    platform_flag = ["Ascend310", "Ascend910", "Hi3796CV300ES", "Ascend710", "Ascend610", "Hi3796CV300CS", "SD3403"]
    base_custom_path = __directory_creation(base_custom_path, "data")
    tune_flag = []
    if "RL" in auto_tiling_mode:
        tune_flag.append("rl")
    if "GA" in auto_tiling_mode:
        tune_flag.append("tiling")

    for tune_path in tune_flag:
        real_path = __directory_creation(base_custom_path, tune_path)
        for soc_version in platform_flag:
            final_path = __directory_creation(real_path, soc_version)
            final_path = __directory_creation(final_path, "custom")
    return True


def _creating_custom_path(job):
    """
    Create custom path
    """
    auto_tiling_mode = job.content["SocInfo"]["autoTilingMode"]
    if "NO_TUNE" in auto_tiling_mode:
        return True

    base_custom_path = job.content["TuneInfo"]["tune_bank_path"]
    tune_bank_flag = True
    if not base_custom_path:
        import auto_tune
        base_custom_path = os.path.dirname(os.path.realpath(auto_tune.__file__))
        base_custom_path = os.path.realpath(os.path.join(base_custom_path, "../../../"))
        tune_bank_flag = False

    if not os.path.isdir(base_custom_path):
        job.error("Check whether the tuning path [{}] exists.".format(base_custom_path))
        return False
    if not os.access(base_custom_path, os.R_OK | os.W_OK | os.X_OK):
        job.error("Check whether the permission on the tuning path [{}] is correct.".format(base_custom_path))
        return False

    if not tune_bank_flag:
        return __creating_default_custom_path(auto_tiling_mode, base_custom_path)
    return True


def _parallel_compilation_init(initialize: TbeJob):
    """
    Tbe parallel compilation initialize
    :param initialize:
    :return:
    """
    os.environ["TE_PARALLEL_COMPILER"] = str(initialize.content["process_num"])
    soc_info = get_soc_info(initialize.content)
    real_debug_level = get_real_op_debug_level(initialize.content)
    auto_tiling_mode = initialize.content["SocInfo"]["autoTilingMode"]
    offline_tune = initialize.content["SocInfo"]["offlineTune"]
    pid_ts = "{}_pid{}".format(datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S%f')[:-3], os.getpid())
    ret = init_multi_process_env(False, soc_info, auto_tiling_mode, real_debug_level,
                                 None, 1, pid_ts)
    if ret is None:
        initialize.error("Init multiprocess env failed")
        return False
    initialize.info("Init multiprocess env success with {} process".format(ret[0]))
    if "RL" in auto_tiling_mode or offline_tune:
        res_queue = ret[1]
        live_checker = ret[2]
        termin_event = ret[3]
        log_level = int(os.getenv("ASCEND_GLOBAL_LOG_LEVEL", "3"))
        from schedule_search.rl_online_tune import rl_tune_init
        ret = rl_tune_init(soc_info, res_queue, live_checker, termin_event, log_level, pid_ts)
        if not ret:
            initialize.error("RL env init failed!")
            return False
        initialize.info("RL Tune init success.")
    if "GA" in auto_tiling_mode:
        start_ga_multi_process(auto_tiling_mode)
        initialize.info("GA Tune init success.")
    return True


def tbe_initialize(job: TbeJob):
    """
    Tbe Initialize
    :param job:
    :return:
    """
    if os.getenv("LD_PRELOAD"):
        os.environ["LD_PRELOAD"] = "libgomp.so.1:" + os.environ["LD_PRELOAD"]
    else:
        os.environ["LD_PRELOAD"] = "libgomp.so.1"
    os.environ["CONTEXT_MODELCOMPILING"] = "TRUE"
    soc_info = get_soc_info(job.content)
    res = te_set_version(*soc_info)
    if not res:
        job.error("Set version failed")
    res = _tune_init(job)
    if not res:
        job.error("Tune init failed")
    lock_file = os.path.join(job.content["SocInfo"]["op_debug_dir"], "kernel_meta", "file.lock")
    local_lock = LocalLock(lock_file)
    try:
        local_lock.lock()
        res = _cann_kb_init(job)
        if res == 1:
            job.error("Cann kb load failed")
        res = _parallel_compilation_init(job)
        if not res:
            job.error("Parallel compilation failed")
    except RuntimeError:
        job.error("Initialize failed with RuntimeError")
    finally:
        local_lock.unlock()
    job.result = "Success"
    return res


def get_auto_tune_support_op_list(job: TbeJob):
    """
    Get GA tune supported op list
    :param job:
    :return:
    """
    from auto_tune_main import enable_auto_tune_support
    auto_tune_op_list = enable_auto_tune_support()
    job.info("auto tune GA support ops list:{}".format(auto_tune_op_list))
    return [x.lower() for x in auto_tune_op_list]


def _normalize_module_name(module_name, py_module_path):
    """
    Normalize module name
    :param module_name:
    :param py_module_path:
    :return:
    """
    if py_module_path not in sys.path:
        sys.path.append(py_module_path)
        sync_syspath(py_module_path)


def check_support(job: TbeJob):
    """
    Check support
    :param job:
    :return:
    """
    op_compute_info_list = get_compute_op_list(job.content)
    if len(op_compute_info_list) != 1:
        job.error("Invalid op compute num ({}) in check_support".format(len(op_compute_info_list)))
        return False
    compute_op_info = op_compute_info_list[0]
    adjust_custom_op_info(compute_op_info)
    inputs, outputs, attrs = assemble_op_args(compute_op_info)
    op_func_name = compute_op_info["func_name"]
    if op_func_name in ("resize_nearest_neighbor_v2_grad_d", "resize_bilinear_v2_grad"):
        attrs.pop(-2)
    op_module_name = get_module_name(compute_op_info)
    _normalize_module_name(op_module_name, compute_op_info["py_module_path"])
    func_name = "check_supported"
    op_type = compute_op_info["type"]
    res = call_op_func((inputs, outputs, attrs), op_module_name, func_name, op_type, {op_type: "high_performance"})
    if isinstance(res, tuple):
        result, reason = res
        result_str = str(result)
    else:
        result_str = str(res)
        reason = None
    if result_str == "True":
        job.result = "FULLY_SUPPORTED"
    elif result_str == "False":
        job.result = "NOT_SUPPORTED"
    elif result_str == "Unknown":
        job.result = "PARTIALLY_SUPPORTED"
        job.info("op module {} check support result is partially supported".format(op_module_name))
    else:
        job.result = "NOT_SUPPORTED"
        job.info("op module {} check support result is {}, not supported".format(op_module_name, result_str))
    if reason:
        job.warning("Unsupported reason is {}".format(reason))
    return True


def select_op_format(job: TbeJob):
    """
    Select op format
    :param job:
    :return:
    """
    compute_op_info_list = get_compute_op_list(job.content)
    if len(compute_op_info_list) != 1:
        job.error("Invalid op compute num ({}) in check_support".format(len(compute_op_info_list)))
        return False
    compute_op_info = compute_op_info_list[0]
    adjust_custom_op_info(compute_op_info)
    inputs, outputs, attrs = assemble_op_args(compute_op_info)
    op_module_name = get_module_name(compute_op_info)
    py_module_path = compute_op_info["py_module_path"]
    _normalize_module_name(op_module_name, py_module_path)
    op_func_name = "op_select_format"
    res = call_op_func((inputs, outputs, attrs), op_module_name, op_func_name)
    job.result = str(res)
    return True


def parallel_pre_compile_op(job: TbeJob):
    """
    Parallel pre compile op
    :param job:
    :return:
    """
    compute_op_info_list = get_compute_op_list(job.content)
    if len(compute_op_info_list) != 1:
        job.error("Invalid op compute num ({}) in pre compile op".format(len(compute_op_info_list)))
        return False
    compute_op_info = compute_op_info_list[0]
    adjust_custom_op_info(compute_op_info)
    _pre_build_compute_op_info(compute_op_info, job)
    return True


def _pre_build_compute_op_info(compute_op, job):
    """
    Prebuild by compute op info
    :param compute_op:
    :param job:
    :return:
    """
    l1_size = job.content["l1_size"]
    if l1_size != -1:
        set_L1_info("op_L1_space", -1)
    inputs, outputs, attrs = assemble_op_args(compute_op, is_single_op_build=True)
    op_module_name = get_module_name(compute_op)
    py_module_path = compute_op["py_module_path"]
    op_func_name = compute_op["func_name"]
    op_type = compute_op["type"]
    op_name = compute_op["op_name"]
    save_op_params(op_name, "prebuild", (outputs, attrs))
    l1_size = job.content["l1_size"]
    set_L1_info("op_L1_space", l1_size)
    _normalize_module_name(op_module_name, py_module_path)
    unknown_shape = compute_op["unknown_shape"]
    is_dynamic_impl = compute_op["is_dynamic_impl"]
    int64_mode = compute_op["int64mode"]
    res = check_op_impl_mode(op_module_name, op_func_name)
    op_impl_mode = job.content["SocInfo"]["op_impl_mode"]
    op_impl_mode_list = job.content["SocInfo"]["op_impl_mode_list"]
    op_full_name = job.content["full_name"]
    if not res:
        if op_impl_mode_list:
            job.warning("The op {} do NOT support op_impl_mode, current op_impl_mode:{}".format(op_type, op_impl_mode))
    else:
        job.info("OpType {} support op_impl_mode, current op_impl_mode:{}".format(op_type, op_impl_mode))
    options = get_options_info(job.content)
    dispatch_prebuild_task(job.source_id, job.id, l1_size, op_module_name, op_full_name,
                           op_type, op_func_name, unknown_shape,
                           (inputs, outputs, attrs, options), int64_mode, is_dynamic_impl,
                           None, job.pass_list)


def get_prebuild_output(op_name):
    """
    get prebuild output
    :param op_name:
    """
    params_str = op_params_to_json(op_name)
    try:
        res = json.loads(params_str)
    except ValueError:
        res = {}
    finally:
        pass
    return res


def do_fuzz_build_tbe_op(job: TbeJob):
    """
    Fuzzy build op
    :param job:
    :return:
    """
    job.result = "NOT_CHANGED"
    return True


def _dump_fusion_op_info_to_json_file(job: TbeJob):
    """
    Dump fusion op info to json file
    :param job:
    :return:
    """
    if not job.sys_para_debug_path or job.sys_para_debug_path == "\0":
        return
    dump_fusion_json(json.dumps(job.content), job.sys_para_debug_path)


def build_single_pre_op(job: TbeJob):
    """
    Build single op
    :param job:
    :return:
    """
    before_build_process(job)
    compute_op_info_list = get_compute_op_list(job.content)
    if len(compute_op_info_list) != 1:
        job.error("Invalid op compute num ({}) in build single op".format(len(compute_op_info_list)))
        return False
    compute_op_info = compute_op_info_list[0]
    adjust_custom_op_info(compute_op_info)
    inputs, outputs, attrs = assemble_op_args(compute_op_info, is_single_op_build=True)
    op_type = compute_op_info["type"]
    l1_size = job.content["l1_size"]
    op_module_name = get_module_name(compute_op_info)
    op_kernel_name = compute_op_info["op_name"]
    py_module_path = compute_op_info["py_module_path"]
    op_name = job.content["full_name"]
    op_func_name = compute_op_info["func_name"]
    _normalize_module_name(op_module_name, py_module_path)
    unknown_shape = compute_op_info["unknown_shape"]
    is_dynamic_impl = compute_op_info["is_dynamic_impl"]
    int64_mode = compute_op_info["int64mode"]
    op_pattern = compute_op_info["pattern"]
    options = get_options_info(job.content)
    fuzz_build_info = get_fuzz_build_info(job.content)
    dispatch_single_op_compile_task(job.source_id, job.id, l1_size, op_module_name, op_name, op_type, op_func_name,
                                    op_kernel_name, unknown_shape, (inputs, outputs, attrs, options), int64_mode,
                                    None, None, is_dynamic_impl, op_pattern,
                                    json.dumps(fuzz_build_info), None, job.pass_list)
    return True


def before_build_process(job: TbeJob):
    """
    Processing before build
    :param job:
    :return:
    """
    l1_size = job.content["l1_size"]
    set_L1_info("op_L1_space", l1_size)
    _dump_fusion_op_info_to_json_file(job)
    offline_tune = job.sys_offline_tune
    if offline_tune:
        dump_fusion_json(json.dumps(job.content), job.sys_tune_dump_path)


def parallel_compile_fusion_op(job: TbeJob):
    """
    Compile fusion op in parallel compiler
    :param job:
    :return:
    """
    l1_size = job.content["l1_size"]
    options = get_options_info(job.content)
    op_kernel_name = job.content["fusion_op_name"]
    op_name = job.content["full_name"]
    relation = ""
    fixpipe_ub_cfg = ""
    dispatch_fusion_op_compile_task(job.source_id, job.id, l1_size, json.dumps(job.content), op_kernel_name, None, None,
                                    options, None, job.pass_list, op_name, relation, fixpipe_ub_cfg, None)
    return True


def ga_tune(job: TbeJob):
    """
    GA tune
    :param job:
    :return:
    """
    l1_size = job.content["l1_size"]
    op_kernel_name = job.content["fusion_op_name"]
    op_name = job.content["full_name"]
    dispatch_autotune_task(job.source_id, job.id, l1_size, json.dumps(job.content), {}, op_kernel_name, op_name)
    job.status = JobStatus.JOB_RUNNING
    return True


def rl_tune_single_op(job: TbeJob):
    """
    RL tune single op
    :param job:
    :return:
    """
    compute_op_info_list = get_compute_op_list(job.content)
    if len(compute_op_info_list) != 1:
        job.error("Invalid op compute num ({}) in rl tune single op".format(len(compute_op_info_list)))
        return False
    compute_op_info = compute_op_info_list[0]
    inputs, outputs, attrs = assemble_op_args(compute_op_info)
    op_type = compute_op_info["type"]
    l1_size = job.content["l1_size"]
    op_module_name = get_module_name(compute_op_info)
    op_kernel_name = compute_op_info["op_name"]
    full_name = compute_op_info["name"]
    py_module_path = compute_op_info["py_module_path"]
    op_func_name = compute_op_info["func_name"]
    _normalize_module_name(op_module_name, py_module_path)
    set_current_op_name(op_kernel_name)
    unknown_shape = compute_op_info["unknown_shape"]
    int64_mode = compute_op_info["int64mode"]
    op_pattern = compute_op_info["pattern"]
    fuzz_build_info = get_fuzz_build_info(job.content)
    auto_tiling_mode = job.content["SocInfo"]["autoTilingMode"]
    device_id = job.content["SocInfo"]["deviceId"]
    options = get_options_info(job.content)
    try:
        build_single_op_from_c(op_module_name, op_func_name, op_type, "build", unknown_shape,
                               (inputs, outputs, attrs), int64_mode, unknown_shape, options,
                               op_pattern, auto_tiling_mode, device_id, json.dumps(fuzz_build_info))
    # pylint: disable=broad-except
    except Exception:
        job.error(
            "Single op {} build failed, no need to do rl tune, json string:{}".format(op_kernel_name, job.json_string))
        exc_type, exc_value, _ = sys.exc_info()
        job.error(
            "exc_type:{}, exc_value:{}, exc_traceback:{}".format(exc_type, exc_value, traceback.format_exc()))
        return False
    finally:
        pass
    tune_op_module_name = op_module_name + "@" + py_module_path
    base_kernel = os.path.join(job.content["SocInfo"]["op_debug_dir"], "kernel_meta", op_kernel_name + ".o")
    from schedule_search.rl_online_tune import dispatch_single_tune_task
    pack_args = pack_op_args(inputs, outputs, attrs)
    res = dispatch_single_tune_task(job.source_id, job.id, l1_size, base_kernel, op_kernel_name, full_name,
                                    tune_op_module_name, op_func_name, op_type, pack_args)
    return _process_rl_tune_result(job, op_type, res)


def rl_tune_fusion_op(job: TbeJob):
    """
    rl tune fusion op
    :param job:
    :return:
    """
    op_kernel_name = job.content["fusion_op_name"]
    set_current_op_name(op_kernel_name)

    try:
        from schedule_search.rl_online_tune import compile_op_by_mp
        compile_op_by_mp(json.dumps(job.content))
    # pylint: disable=broad-except
    except Exception:
        job.error(
            "Fusion op {} build failed, no need to do rl tune, json string:{}".format(op_kernel_name, job.json_string))
        exc_type, exc_value, _ = sys.exc_info()
        job.error(
            "exc_type:{}, exc_value:{}, exc_traceback:{}".format(exc_type, exc_value, traceback.format_exc()))
        return False
    finally:
        pass
    l1_size = job.content["l1_size"]
    base_kernel = os.path.join(job.content["SocInfo"]["op_debug_dir"], "kernel_meta", op_kernel_name + ".o")
    compute_op_list = get_compute_op_list(job.content)
    op_module_names_str = ""
    op_type_set = set()
    for op in compute_op_list:
        op_module_names_str = ','.join([op_module_names_str, get_module_name(op)])
        op_type_set.add(op["type"])
    op_module_names_str = op_module_names_str[1:]
    op_type = "__".join(list(op_type_set))
    from schedule_search.rl_online_tune import dispatch_fusion_tune_task
    res = dispatch_fusion_tune_task(job.source_id, job.id, l1_size, base_kernel, op_kernel_name, op_module_names_str,
                                    json.dumps(job.content))
    return _process_rl_tune_result(job, op_type, res)


def _process_rl_tune_result(job, op_type, res):
    if not res:
        from schedule_search.tune_util import filter_black_op_type
        res = bool(job.sys_offline_tune or os.getenv("REPEAT_TUNE", "False").lower() != "true" or filter_black_op_type(
            op_type))
    else:
        job.status = JobStatus.JOB_RUNNING
        res = True
    return res


def get_finish_tasks(source_id):
    """
    Get finish task from parallel compilation framework
    :return task info list
    """
    return get_finished_compilation_task(source_id)


def tbe_finalize(auto_tiling_mode, offline_tune, job: TbeJob):
    """
    finalize tbe parallel compilation resource
    :param auto_tiling_mode: RL/GA/RL,GA
    :param offline_tune: True/False
    :param job: TbeJob
    :return: None
    """
    deinit_multi_process_env()
    if "RL" in auto_tiling_mode or offline_tune:
        from schedule_search.rl_online_tune import rl_tune_deinit
        rl_tune_deinit()
    res = _cann_kb_finalize()
    if res == 1:
        job.error("Cann kb unload failed")
        return False
    clear_fusion_params()
    _remove_cache(job)
    return True

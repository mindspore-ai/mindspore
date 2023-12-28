# Copyright 2021-2023 Huawei Technologies Co., Ltd
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
from datetime import datetime, timezone

from tbe.common.repository_manager.interface import cann_kb_finalize, cann_kb_init
from te.platform.cce_conf import te_set_version
from te.platform.cce_policy import set_L1_info
from te_fusion.compile_task_manager import dispatch_prebuild_task, dispatch_single_op_compile_task, \
    dispatch_fusion_op_compile_task
from te_fusion.compile_task_manager import sync_syspath
from te_fusion.fusion_manager import call_op_func, clear_fusion_params, check_op_impl_mode, \
    save_op_params, op_params_to_json
from te_fusion.fusion_util import dump_fusion_json
from te_fusion.parallel_compilation import init_multi_process_env, deinit_multi_process_env, \
    get_finished_compilation_task

from .tbe_helper import get_soc_info, assemble_op_args, get_compute_op_list, get_options_info, get_context_param, \
    get_fuzz_build_info, adjust_custom_op_info, get_module_name, get_real_op_debug_level, LocalLock
from .tbe_job import TbeJob


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
        os.makedirs(path, 0o700)
    return path


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
    pid_ts = "{}_pid{}".format(datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S%f')[:-3], os.getpid())
    ret = init_multi_process_env(False, soc_info, auto_tiling_mode, real_debug_level,
                                 None, 1, pid_ts)
    if ret is None:
        initialize.error("Init multiprocess env failed")
        return False
    initialize.info("Init multiprocess env success with {} process".format(ret[0]))
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
    op_impl_mode = compute_op_info["op_impl_mode"]
    res = call_op_func((inputs, outputs, attrs), op_module_name, func_name, op_type, op_impl_mode)
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
    op_impl_mode = compute_op_info["op_impl_mode"]
    _normalize_module_name(op_module_name, py_module_path)
    op_func_name = "op_select_format"
    res = call_op_func((inputs, outputs, attrs), op_module_name, op_func_name, op_impl_mode)
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
    op_impl_mode = compute_op["op_impl_mode"]
    op_full_name = job.content["full_name"]
    if not res:
        if op_impl_mode:
            job.warning("The op {} do NOT support op_impl_mode, current op_impl_mode:{}".format(op_type, op_impl_mode))
    else:
        job.info("OpType {} support op_impl_mode, current op_impl_mode:{}".format(op_type, op_impl_mode))
    options = get_options_info(job.content)
    context_param = get_context_param()
    dispatch_prebuild_task(job.source_id, job.id, l1_size, op_module_name, op_full_name,
                           op_type, op_func_name, unknown_shape,
                           (inputs, outputs, attrs, options), int64_mode, is_dynamic_impl,
                           context_param, job.pass_list, op_impl_mode=op_impl_mode)


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
    op_impl_mode = compute_op_info["op_impl_mode"]
    options = get_options_info(job.content)
    fuzz_build_info = get_fuzz_build_info(job.content)
    dispatch_single_op_compile_task(job.source_id, job.id, l1_size, op_module_name, op_name, op_type, op_func_name,
                                    op_kernel_name, unknown_shape, (inputs, outputs, attrs, options), int64_mode,
                                    None, None, is_dynamic_impl, op_pattern,
                                    json.dumps(fuzz_build_info), None, job.pass_list, op_impl_mode=op_impl_mode)
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
        pass
    res = _cann_kb_finalize()
    if res == 1:
        job.error("Cann kb unload failed")
        return False
    clear_fusion_params()
    _remove_cache(job)
    return True

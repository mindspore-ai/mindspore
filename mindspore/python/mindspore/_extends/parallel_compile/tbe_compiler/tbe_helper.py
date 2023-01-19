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
"""tbe helper to parse json content"""
from __future__ import absolute_import
import os
import stat
import fcntl
from enum import Enum

from .tbe_job import JobType


def create_dir(dir_path):
    """
    create dir
    :param dir_path:
    :return: T/F
    """
    dir_path = os.path.realpath(dir_path)
    is_exists = os.path.exists(dir_path)
    if not is_exists:
        try:
            os.makedirs(dir_path, 0o750, exist_ok=True)
        except (OSError, TypeError) as excep:
            raise excep
        finally:
            pass
    return True


def write_to_file(file_path, content=""):
    """
    write to file
    :param file_path:
    :param content:
    :return: T/F
    """
    dir_name = os.path.dirname(file_path)
    ret = create_dir(dir_name)
    if not ret:
        return False

    with os.fdopen(os.open(file_path, os.O_WRONLY | os.O_CREAT, \
        stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP), 'w') as file_handler:
        file_handler.write(content)
    return True


class LocalLock:
    """
    LocalLock
    """

    def __init__(self, lock_file):
        if not os.path.exists(lock_file):
            write_to_file(lock_file)
        self.lock_fd = os.open(lock_file, os.O_WRONLY | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP)

    def __del__(self):
        try:
            os.close(self.lock_fd)
        except OSError:
            pass
        finally:
            pass

    def lock(self):
        """
        lock
        """
        fcntl.flock(self.lock_fd, fcntl.LOCK_EX)

    def unlock(self):
        """
        unlock
        """
        fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
        os.close(self.lock_fd)


class BuildType(Enum):
    """ Build Type """
    INITIALLY = "initially_build"
    FUZZILY = "fuzzily_build"
    ACCURATELY = "accurately"


def check_job_json(job_info):
    """
    Check tne compilation job json's required element
    :param job_info:tne compilation job json
    :return: raise value error if wrong
    """
    job_type_list = [job_type.value for _, job_type in JobType.__members__.items()]
    if 'source_id' not in job_info:
        raise ValueError("Json string Errors, key:source_id not found.")
    if 'job_id' not in job_info:
        raise ValueError("Json string Errors, key:job_id not found.")
    if 'job_type' not in job_info or not job_info['job_type']:
        raise ValueError("Json string Errors, key:job_type not found.")
    if job_info['job_type'] not in job_type_list:
        raise ValueError("Invalid job type: {}.".format(job_info['job_type']))
    if 'job_content' not in job_info:
        raise ValueError("Json string Errors, key:job_content not found.")


def reset_op_debug_level_in_soc_info(level):
    """
    :param level: op_debug_level, if level is 3 or 4, replace it with 0
    :return: op_debug_level
    """
    if level in ("3", "4"):
        level = "0"
    return level


def get_real_op_debug_level(initialize_job_info):
    """
    :param initialize_job_info: initialize_job_info
    :return: origin op_debug_level for init_multi_process_env
    """
    return initialize_job_info["SocInfo"]["op_debug_level"]


def get_soc_info(initialize_job_info):
    """
    Get soc info from initialize job info
    :param initialize_job_info:
    :return: soc info
    """
    soc_param = dict()
    soc_param["op_impl_mode"] = initialize_job_info["SocInfo"]["op_impl_mode"]
    soc_param["op_debug_level"] = reset_op_debug_level_in_soc_info(initialize_job_info["SocInfo"]["op_debug_level"])
    soc_param["op_debug_config"] = initialize_job_info["SocInfo"]["op_debug_config"]
    soc_param["op_impl_mode_list"] = initialize_job_info["SocInfo"]["op_impl_mode_list"]
    soc_param["op_debug_dir"] = initialize_job_info["SocInfo"]["op_debug_dir"]
    soc_param["vector_fp_ceiling"] = initialize_job_info["SocInfo"]["vector_fp_ceiling"]
    soc_param['mdl_bank_path'] = initialize_job_info["SocInfo"]["mdl_bank_path"]
    soc_param['te_version'] = initialize_job_info["SocInfo"]["te_version"]
    soc_param['op_bank_path'] = initialize_job_info["SocInfo"]["op_bank_path"]
    soc_param['kernel_meta_temp_dir'] = initialize_job_info["SocInfo"]["kernel_meta_temp_dir"]

    soc_info = list()
    soc_info.append(initialize_job_info["SocInfo"]["socVersion"])
    soc_info.append(initialize_job_info["SocInfo"]["coreType"])
    soc_info.append(initialize_job_info["SocInfo"]["coreNum"])
    soc_info.append(initialize_job_info["SocInfo"]["l1Fusion"])
    soc_info.append(initialize_job_info["SocInfo"]["l2Mode"])
    soc_info.append(initialize_job_info["SocInfo"]["l2Fusion"])
    soc_info.append(soc_param)

    return soc_info


def check_arg_info(io_info):
    """
    Check parameter Validity.
    :param io_info:A dict, to be checked.
    :return: Exception: If specific keyword is not found.
    """
    if 'shape' not in io_info:
        raise ValueError("Json string Errors, key:shape not found.")
    if 'ori_shape' not in io_info:
        raise ValueError("Json string Errors, key:ori_shape not found.")
    if 'format' not in io_info or not io_info['format']:
        raise ValueError("Json string Errors, key:format not found.")
    if 'ori_format' not in io_info or not io_info['ori_format']:
        raise ValueError("Json string Errors, key:ori_format not found.")
    if 'dtype' not in io_info or not io_info['dtype']:
        raise ValueError("Json string Errors, key:dtype not found.")
    if 'param_type' not in io_info or not io_info['param_type']:
        raise ValueError("Json string Errors, key:param_type not found.")


def get_input_output_args(io_info):
    """
    Get input/output args from io info
    :param io_info:
    :return:input/output args
    """
    args = []
    if io_info is None:
        return args
    for item in io_info:
        if isinstance(item, dict):
            arg = get_single_io_arg(item)
            args.append(arg)
        elif isinstance(item, list):
            dyn_arg = []
            for info in item:
                arg = get_single_io_arg(info)
                dyn_arg.append(arg)
            args.append(tuple(dyn_arg))
    return args


def __dynamic_range_process(info):
    """
    Current process need move to tbe json creator.
    """
    if 'range' in info:
        for i in range(len(info['range'])):
            if info['range'][i][1] == -1:
                info['range'][i][1] = None
    return info


def get_single_io_arg(info):
    """
    Get single input/output arg from io info
    :param info:
    :return:input/output arg
    """
    if 'valid' not in info:
        raise ValueError("Json string Errors, key:valid not found.")
    res = None
    if info['valid']:
        check_arg_info(info)
        del info['valid']
        del info['name']
        res = info
    if not res:
        return res
    return __dynamic_range_process(res)


def assemble_op_args(compute_op_info, is_single_op_build=False):
    """
    Assemble op args
    :param compute_op_info:
    :param is_single_op_build: is used for single op build or not
    :return: op args
    """
    inputs_info = compute_op_info["input_desc"] if "input_desc" in compute_op_info.keys() else None
    outputs_info = compute_op_info["output_desc"] if "output_desc" in compute_op_info.keys() else None
    if is_single_op_build:
        attrs = []
        attrs_info = compute_op_info["attrs"] if "attrs" in compute_op_info.keys() else []
        for item in attrs_info:
            if item["valid"]:
                attrs.append(item)
    else:
        attrs = compute_op_info["attr_desc"] if "attr_desc" in compute_op_info.keys() else []
    inputs = get_input_output_args(inputs_info)
    outputs = get_input_output_args(outputs_info)
    attrs.append(compute_op_info["op_name"])
    return inputs, outputs, attrs


def get_compute_op_list(job_content):
    """
    Get compute op info list from job content info
    :param job_content: tbe compilation content info
    :return: compute op info list
    """
    op_list = job_content["op_list"]
    op_compute_list = []
    for op in op_list:
        if op["type"] != "Data":
            op_compute_list.append(op)
    return op_compute_list


def get_options_info(job_content):
    """
    Get options info
    :param job_content:
    :return: options
    """
    options = dict()
    options["socVersion"] = job_content["SocInfo"]["socVersion"]
    options["coreType"] = job_content["SocInfo"]["coreType"]
    options["coreNum"] = job_content["SocInfo"]["coreNum"]
    options["l1Fusion"] = job_content["SocInfo"]["l1Fusion"]
    options["l2Fusion"] = job_content["SocInfo"]["l2Fusion"]
    options["l2Mode"] = job_content["SocInfo"]["l2Mode"]
    options["op_debug_level"] = reset_op_debug_level_in_soc_info(job_content["SocInfo"]["op_debug_level"])
    options["op_debug_config"] = job_content["SocInfo"]["op_debug_config"]
    options["op_impl_mode"] = job_content["SocInfo"]["op_impl_mode"]
    options["op_debug_dir"] = job_content["SocInfo"]["op_debug_dir"]
    options["mdl_bank_path"] = job_content["SocInfo"]["mdl_bank_path"]
    options["te_version"] = job_content["SocInfo"]["te_version"]
    options["op_bank_path"] = job_content["SocInfo"]["op_bank_path"]
    options["deviceId"] = job_content["SocInfo"]["deviceId"]
    options["autoTilingMode"] = job_content["SocInfo"]["autoTilingMode"]
    options["op_impl_mode_list"] = job_content["SocInfo"]["op_impl_mode_list"]
    options["kernel_meta_temp_dir"] = job_content["SocInfo"]["kernel_meta_temp_dir"]
    options["deterministic"] = job_content["SocInfo"]["deterministic"]
    options["status_check"] = "false"
    return options


def get_fuzz_build_info(job_content):
    """
    Get fuzz build info from job content info
    :param job_content: job content info
    :return: fuzz build info
    """
    op_compute_info = get_compute_op_list(job_content)[0]
    fuzz_build_info = dict()
    fuzz_build_info["compile_type"] = "accurately_build"
    if op_compute_info["build_type"] == BuildType.FUZZILY.value:
        fuzz_build_info["compile_type"] = "fuzzily_build"
    fuzz_build_info["miss_support_info"] = op_compute_info["miss_support_info"]
    fuzz_build_info["max_kernel_id"] = op_compute_info["max_kernel_id"]
    json_path = os.path.join(job_content["SocInfo"]["op_debug_dir"], "kernel_meta", op_compute_info["name"] + ".json")
    fuzz_build_info["incremental_link"] = ""
    if op_compute_info["build_type"] == BuildType.FUZZILY.value:
        fuzz_build_info["incremental_link"] = os.path.realpath(json_path)
    return fuzz_build_info


def get_func_names(job_content):
    """
    Get function names from job content json
    :param job_content: job content info
    :return: function names
    """
    func_names = []
    for op in job_content["op_list"]:
        if "func_name" in op:
            func_names.append(op["func_name"])
    return func_names


def get_module_name(compute_op_info):
    """
    get compute_op_info
    :param compute_op_info:
    :return:
    """
    is_dynamic_impl = compute_op_info["is_dynamic_impl"]
    unknown_shape = compute_op_info["unknown_shape"]
    op_module_name = compute_op_info["module_name"]
    if is_dynamic_impl or unknown_shape:
        d = ".dynamic."
        op_module_name = d.join((op_module_name.split(".")[0], op_module_name.split(".")[-1]))
    return op_module_name


def adjust_custom_op_info(compute_op_info):
    """
    adjust custom op info
    :param compute_op_info:
    :return:
    """
    py_module_path = compute_op_info["py_module_path"]
    if os.path.isfile(py_module_path):
        py_module_path, file_name = os.path.split(py_module_path)
        module_name, _ = os.path.splitext(file_name)
        compute_op_info["py_module_path"] = py_module_path
        compute_op_info["module_name"] = module_name


def pack_op_args(inputs, outputs, attrs):
    """
    flatten inputs outputs attrs
    """
    op_args = (inputs, outputs, attrs)
    return [item for arg in op_args for item in arg]

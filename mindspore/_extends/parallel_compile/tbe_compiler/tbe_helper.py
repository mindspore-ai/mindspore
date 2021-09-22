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
import os
from enum import Enum

from .tbe_job import JobType


class BuildType(Enum):
    """ Build Type """
    INITIALLY = "initially_build"
    FUZZILY = "fuzzily_build"
    ACCURATELY = "accurately"


job_type_list = [job_type.value for _, job_type in JobType.__members__.items()]


def check_job_json(job_info):
    """
    Check tne compilation job json's required element
    :param job_info:tne compilation job json
    :return: raise value error if wrong
    """
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


def get_soc_info(initialize_job_info):
    """
    Get soc info from initialize job info
    :param initialize_job_info:
    :return: soc info
    """
    soc_param = dict()
    soc_param["op_impl_mode"] = initialize_job_info["SocInfo"]["op_impl_mode"]
    soc_param["op_debug_level"] = initialize_job_info["SocInfo"]["op_debug_level"]
    soc_param["op_impl_mode_list"] = initialize_job_info["SocInfo"]["op_impl_mode_list"]
    soc_param["op_debug_dir"] = initialize_job_info["SocInfo"]["op_debug_dir"]
    soc_param["vector_fp_ceiling"] = initialize_job_info["SocInfo"]["vector_fp_ceiling"]
    soc_param['mdl_bank_path'] = initialize_job_info["SocInfo"]["mdl_bank_path"]
    soc_param['op_bank_path'] = initialize_job_info["SocInfo"]["op_bank_path"]

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


def get_single_io_arg(info):
    """
    Get single input/output arg from io info
    :param info:
    :return:input/output arg
    """
    if 'valid' not in info:
        raise ValueError("Json string Errors, key:valid not found.")
    if info['valid']:
        check_arg_info(info)
        del info['valid']
        del info['name']
        res = info
    else:
        res = None
    return res


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
            if item["valid"] and item["name"] != "isRef":
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
    options["op_debug_level"] = job_content["SocInfo"]["op_debug_level"]
    options["op_impl_mode"] = job_content["SocInfo"]["op_impl_mode"]
    options["op_debug_dir"] = job_content["SocInfo"]["op_debug_dir"]
    options["mdl_bank_path"] = job_content["SocInfo"]["op_debug_level"]
    options["op_bank_path"] = job_content["SocInfo"]["op_bank_path"]
    options["deviceId"] = job_content["SocInfo"]["deviceId"]
    options["autoTilingMode"] = job_content["SocInfo"]["autoTilingMode"]
    options["op_impl_mode_list"] = job_content["SocInfo"]["op_impl_mode_list"]
    return options


def get_fuzz_build_info(job_content):
    """
    Get fuzz build info from job content info
    :param job_content: job content info
    :return: fuzz build info
    """
    op_compute_info = get_compute_op_list(job_content)[0]
    fuzz_build_info = dict()
    fuzz_build_info["compile_type"] = "fuzzily_build" if op_compute_info["build_type"] == BuildType.FUZZILY.value \
        else "accurately_build"
    fuzz_build_info["miss_support_info"] = op_compute_info["miss_support_info"]
    fuzz_build_info["max_kernel_id"] = op_compute_info["max_kernel_id"]
    fuzz_build_info["incremental_link"] = os.path.realpath(
        job_content["SocInfo"]["op_debug_dir"] + "/kernel_meta/" + op_compute_info["name"] + ".json") if \
        op_compute_info["build_type"] == BuildType.FUZZILY.value else ""
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
    dynamic_compile_static = compute_op_info["dynamic_compile_static"]
    unknown_shape = compute_op_info["unknown_shape"]
    op_module_name = compute_op_info["module_name"]
    if dynamic_compile_static or unknown_shape:
        op_module_name = op_module_name.split(".")[0] + ".dynamic." + op_module_name.split(".")[-1]
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

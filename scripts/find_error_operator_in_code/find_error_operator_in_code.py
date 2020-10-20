#!/usr/bin/env python3
# coding=UTF-8
# Copyright 2020 Huawei Technologies Co., Ltd
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
"""find error operator in code"""

import os
import json
import argparse


# step 1: get error_full_name, error_kernel_name, error_context, and trace_back_log.
def get_log_error_and_traceback(log_file):
    """
    get error_full_name, error_kernel_name, error_context, and trace_back_log.
    """
    with open(log_file, 'r+') as log:
        log_context_list = log.readlines()

    # get error log, error full_name, kernel_name, stream_id and task_id in log
    task_stream_log = []
    error_context = ""
    error_stream_id = ""
    error_task_id = ""
    error_kernel_name = ""
    for log_context in log_context_list:
        if "stream_id" in log_context and "task_id" in log_context:
            task_stream_log.append(log_context)
        if "[ERROR]" not in log_context:
            continue
        error_context = error_context + log_context
        if "fault kernel_name" in log_context and "stream_id" in log_context and "task_id" in log_context:
            error_stream_id = log_context.split("stream_id")[-1].split(",")[0].split("=")[-1]
            error_task_id = log_context.split("task_id")[-1].split(",")[0].split("=")[-1]
            error_kernel_name = log_context.split("kernel_name")[-1].split(",")[0].split("=")[-1]

    # get full name
    error_full_name = ""
    for task_stream in task_stream_log:
        if "[ERROR]" in task_stream:
            continue
        if "stream_id:" + error_stream_id in task_stream and "task_id:" + error_task_id in task_stream:
            error_full_name = task_stream.split("ask name:")[-1].split(" ")[0]
    if error_full_name == "":
        exit("[ERROR] Find full name in task exception need to "
             "set the log level to warning at least [export GLOG_v=2].")
    # get Traceback log
    log_context_str = "".join(log_context_list)
    trace_back_log = "Traceback" + log_context_str.split("Traceback")[-1]

    return error_full_name, error_kernel_name, error_context, trace_back_log


# step 2.1: get error operator shape and dytpe in kernel_meta.
def get_shape_dtyep_in_kernel_meta(log_file_path, kernel_meta):
    """
    get error operator shape and dytpe in kernel_meta.
    """
    error_full_name, error_kernel_name, _, _ = get_log_error_and_traceback(log_file_path)
    # the hash value of error kernel name
    oper_info_dict = {}
    error_kernel_name_file = error_kernel_name.split("__kernel")[0] + ".info"
    error_kernel_name_file_path = os.path.join(kernel_meta, error_kernel_name_file)
    with open(error_kernel_name_file_path, 'r+') as kernel_info:
        kernel_info_str = kernel_info.readline()
    kernel_info_json = json.loads(kernel_info_str)
    oper_info_dict["full_name"] = error_full_name
    oper_info_dict["inputs"] = kernel_info_json["inputs"]
    oper_info_dict["input_num"] = len(kernel_info_json["inputs"])
    oper_info_dict["outputs"] = kernel_info_json["outputs"]
    oper_info_dict["output_num"] = len(oper_info_dict["outputs"])
    return oper_info_dict


# step 2.2: get error operator shape and dytpe in tbe compile.
def get_shape_dtyep_in_tbe_compile(tbe_compile_log):
    """
    get error operator shape and dytpe in tbe compile.
    """
    oper_info = ""
    oper_info_dict = {}
    with open(tbe_compile_log, 'r+') as log:
        log_context_list = log.readlines()
    for compile_log in log_context_list:
        if compile_log.startswith("input_args:"):
            oper_info = compile_log.split("input_args:")[-1]
    oper_info_json = json.loads(oper_info)
    oper_info_dict["full_name"] = oper_info_json["full_name"]
    oper_info_dict["inputs"] = oper_info_json["op_info"]["inputs"]
    oper_info_dict["input_num"] = len(oper_info_dict["inputs"])
    oper_info_dict["outputs"] = oper_info_json["op_info"]["outputs"]
    oper_info_dict["output_num"] = len(oper_info_dict["outputs"])
    return oper_info_dict


# step 3: show full name, input_ouput_atts info and the line in source code.
def read_all_source_code(full_name):
    """
    research operation in source code
    """
    code_path = input("[WARNING] Do you want to research in source code? set source code path to research or "
                      "press enter to research in current path, input n/no to exit.\nInput: ")
    if code_path.lower() == "n" or code_path.lower() == "no":
        exit("[INFO] Exit.")
    code_path = code_path
    if code_path == '':
        code_path = os.path.abspath(os.path.dirname(__file__))
    print("[INFO] Research in {}".format(code_path))
    current_script_file = os.path.basename(__file__)
    last_scope = full_name.split("/")[-2]
    oper_name = full_name.split("/")[-1]
    find_name = last_scope if last_scope not in ("Default", "network") else oper_name
    source_code_file = []
    for code_path, _, file_list in os.walk(code_path):
        for file_name in file_list:
            if file_name.endswith(".py") and file_name != current_script_file:
                code_file_path = os.path.join(code_path, file_name)
                source_code_file.append(code_file_path)
    find_code_flag = False
    for code_file_path in source_code_file:
        with open(code_file_path, 'r+') as code:
            code_context_list = code.readlines()
        for line, code_context in enumerate(code_context_list):
            if find_name in code_context:
                find_code_flag = True
                print("[INFO] find in {}\nLine: {}\nCode: {}\n".format(code_file_path, line + 1, code_context))

    if not find_code_flag:
        exit("[WARNING] Cannot find error operation in source code!")


def dtype_map(tensor_dtype, oper_name):
    """
    type map dict
    """
    dtype_dict = {"F8": "float8", "F16": "float16", "F32": "float32", "F64": "float64",
                  "I8": "int8", "I16": "int16", "I32": "int32", "I64": "int64",
                  "U8": "uint8", "U16": "uint16", "U32": "uint32", "U64": "uint64",
                  "TypeType": "TypeType", "Bool": "bool"}
    if tensor_dtype not in dtype_dict:
        if len(tensor_dtype) <= 10:
            print("[WARNING] Type convert fail! {} not in dtype map in operator {}.".format(tensor_dtype, oper_name))
        return tensor_dtype
    return dtype_dict[tensor_dtype]


def operator_check(operator_name):
    """
    check operator name
    """
    block_list = ["tuple_getitem", "make_tuple", "Depend", "GetNext"]
    return operator_name in block_list


def convert_validate_operator_info(input_tensor_str, oper_name, val_context):
    """
    convert operator info of validate.dat file
    """
    tensor_dtype, tensor_shape = "", ""
    if "Tuple" in input_tensor_str:
        pass
    elif "Ref[" in input_tensor_str:
        tensor_dtype = dtype_map(input_tensor_str.split("sor(")[-1].split(")][")[0], oper_name)
        tensor_shape = "[" + input_tensor_str.split(")][")[-1].split("]")[0] + "]"
    else:
        tensor_dtype = dtype_map(input_tensor_str.split("sor(")[-1].split(")[")[0], oper_name)
        tensor_shape = "[" + input_tensor_str.split(")[")[-1].split("]")[0] + "]"
    if tensor_dtype == "TypeType":
        tensor_dtype = dtype_map(val_context.split("DstT=")[-1].split(", ")[0], oper_name)
    return tensor_dtype, tensor_shape


def get_operator_info(val_context, input_info_dict):
    """
    get full name, input shape and type from validate file
    """
    if "PrimitivePy::" in val_context:
        oper_name = val_context.split("PrimitivePy::")[-1].split("@")[0].strip()
    elif "Primitive" in val_context:
        oper_name = val_context.split("Primitive::")[-1].split("(%")[0].strip()
    scope_name = val_context.split("#scope:")[-1].split("#")[0].strip()
    input_info = val_context.split("#scope:")[0].split("#")[-1].strip()
    full_name = scope_name + "/" + oper_name
    if "Tuple" in input_info:
        input_info_dict["tuple_input"] = input_info
        input_info_dict["tuple_input_name"] = full_name
    input_info_list = input_info.lstrip("(").rstrip(")").split(", T")
    input_info_dict["input_num"] = len(input_info_list)

    output_info = val_context.split(" = Primitive")[0].split(":")[-1].strip()
    if "Tuple" in output_info:
        input_info_dict["tuple_output"] = output_info
        input_info_dict["tuple_output_name"] = full_name
    output_info_list = output_info.lstrip("(").rstrip(")").split(", T")
    input_info_dict["output_num"] = len(output_info_list)
    if not operator_check(oper_name):
        for i, input_tensor_str in enumerate(input_info_list):
            if i > 0: input_tensor_str = "T" + input_tensor_str
            tensor_dtype, tensor_shape = convert_validate_operator_info(input_tensor_str, oper_name, val_context)
            input_info_dict["input_" + str(i)] = [tensor_dtype, tensor_shape]
        for i, output_tensor_str in enumerate(output_info_list):
            if i > 0: output_tensor_str = "T" + output_tensor_str
            tensor_dtype, tensor_shape = convert_validate_operator_info(output_tensor_str, oper_name, val_context)
            input_info_dict["output_" + str(i)] = [tensor_dtype, tensor_shape]

    input_info_dict["full_name"] = full_name
    return input_info_dict


def py_pre_ad_parse(val_file):
    """
    parse the validate file and get code line
    """
    oper_start_info_num = 0
    oper_end_char_flag = False
    oper_start_char_flag = False
    input_dict = {}
    with open(val_file, 'r+') as val:
        val_context_list = val.readlines()
    input_info_dict = {}
    for val_context in val_context_list:
        val_context = val_context.strip()
        if val_context.startswith(") {"):
            oper_start_char_flag = True
            oper_end_char_flag = False

        if val_context == "}":
            oper_end_char_flag = True

        oper_start_info_flag = False
        if val_context.startswith("%") and val_context[1].isdigit():
            input_info_dict = {}
            oper_start_info_num += 1
            oper_start_info_flag = True

        if oper_start_char_flag and not oper_end_char_flag:
            if oper_start_info_num >= 1 and oper_start_info_flag:
                if "PrimitivePy::" in val_context or "Primitive" in val_context:
                    input_info_dict = get_operator_info(val_context, input_info_dict)
            if "In file" not in val_context or "in_file" in input_info_dict:
                continue
            input_info_dict["in_file"] = val_context.split("#")[-1].strip().rstrip("/")
            input_dict[oper_start_info_num] = input_info_dict

    return input_dict


def show_input_output_info(kernel_info_json, show_name):
    """
    print input or ouput info to stdout
    """
    if show_name + "_num" in kernel_info_json:
        print("[INFO] Have {} {} in operator:".format(kernel_info_json[show_name + "_num"], show_name))
        for i in range(kernel_info_json[show_name + "_num"]):
            for j in range(len(kernel_info_json[show_name + "s"][i])):
                if "ori_shape" not in kernel_info_json[show_name + "s"][i][j] and "dtype" not in \
                        kernel_info_json[show_name + "s"][i][j]:
                    print("       {} {}/{}th: {}.".format(show_name, j + 1, i + 1,
                                                          kernel_info_json[show_name + "s"][i][j]))
                else:
                    kernel_dtype = kernel_info_json[show_name + "s"][i][j].setdefault("dtype", None)
                    kernel_shape = str(kernel_info_json[show_name + "s"][i][j].setdefault("ori_shape", None))
                    print("       {} {}/{}th: dtype is {}, shape is {}.".format(show_name, j + 1, i + 1,
                                                                                kernel_dtype, kernel_shape))


def show_error_operator_info(val_file, kernel_info_json):
    """
    find error operator
    """
    error_full_name = kernel_info_json["full_name"].split("-op")[0]
    input_dict = py_pre_ad_parse(val_file)
    find_num = 0
    for _, oper_info in input_dict.items():
        if oper_info["full_name"] != error_full_name:
            continue

        if kernel_info_json["input_num"] > oper_info["input_num"] \
                or kernel_info_json["output_num"] > oper_info["output_num"]:
            continue

        find_oper_input = True
        for i in range(kernel_info_json["input_num"]):
            for j in range(len(kernel_info_json["inputs"][i])):
                kernel_dtype = kernel_info_json["inputs"][i][j].setdefault("dtype", None)
                kernel_shape = str(kernel_info_json["inputs"][i][j].setdefault("ori_shape", None))
                validate_dtype = oper_info["input_" + str(i)][0]
                validate_shape = oper_info["input_" + str(i)][1]
                if kernel_shape != validate_shape or kernel_dtype != validate_dtype:
                    find_oper_input = False

        find_oper_output = True
        for i in range(kernel_info_json["output_num"]):
            for j in range(len(kernel_info_json["outputs"][i])):
                kernel_dtype = kernel_info_json["outputs"][i][j].setdefault("dtype", None)
                kernel_shape = str(kernel_info_json["outputs"][i][j].setdefault("ori_shape", None))
                validate_dtype = oper_info["output_" + str(i)][0]
                validate_shape = oper_info["output_" + str(i)][1]
                if kernel_shape != validate_shape or kernel_dtype != validate_dtype:
                    find_oper_output = False

        if find_oper_input or find_oper_output:
            find_num += 1
            print("[INFO] Find operation {} times!".format(find_num))
            print("[INFO] {}".format(oper_info["in_file"]))
            print("[INFO] Exception operator is \"{}\".".format(error_full_name))

            if not find_oper_input:
                print("[WARNING] Cannot match input information! Please check whether the operator's input is:")
            show_input_output_info(kernel_info_json, "input")
            if not find_oper_output:
                print("[WARNING] Cannot match output information! Please check whether the operator's output is:")
            show_input_output_info(kernel_info_json, "output")

    if find_num == 0:
        print("[WARNING] Cannot find operation! Need to find in the script based on the following information:")
        print("[INFO] Exception operator full name is \"{}\".".format(error_full_name))
        show_input_output_info(kernel_info_json, "input")
        show_input_output_info(kernel_info_json, "output")
        read_all_source_code(error_full_name)


def get_task_type(log_file):
    """
    get the error type
    """
    with open(log_file, 'r+') as log:
        log_context_list = log.readlines()
    for log_context in log_context_list:
        if "PreCompileProcessFailed" in log_context:
            return "compile"
        if "run task error" in log_context:
            return "exception"
    return None


def start_find(log_file_path, code_line_file_path, kernel_meta_path):
    """
    start find error operation in code.
    """
    task = get_task_type(log_file_path)
    if task == "exception":
        print("[INFO] Detect \"task exception error\".")
        kernel_info_json = get_shape_dtyep_in_kernel_meta(log_file_path, kernel_meta_path)
        show_error_operator_info(code_line_file_path, kernel_info_json)
    elif task == "compile":
        print("[INFO] Detect \"compile error\".")
        the_kernel_json = get_shape_dtyep_in_tbe_compile(log_file_path)
        show_error_operator_info(code_line_file_path, the_kernel_json)
    else:
        exit("[ERROR] Currently only support task exception or tbe compile error!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show error info')
    parser.add_argument('--log_path', '-p', type=str.lower, default="", help='log file path (option)')
    parser.add_argument('--log_name', '-n', type=str.lower, default="", required=True, help='log file name')
    args_opt = parser.parse_args()
    current_path = args_opt.log_path
    if current_path == "":
        print("[WARNING] No log_path parameter, use current path as log path.")
        current_path = os.path.abspath(os.path.dirname(__file__))

    log_name_file = os.path.join(current_path, args_opt.log_name)
    kernel_meta_file = os.path.join(current_path, "kernel_meta")

    code_line_file = ""
    for filename in os.listdir(os.path.join(current_path)):
        if "_validate.dat" in filename:
            code_line_file = filename
        if "_py_pre_ad.dat" in filename:
            code_line_file = filename
    if code_line_file == "":
        exit("[ERROR] Please set \"save_graphs=True\" in context to save py_pre_ad.dat or validate.dat file!")
    code_line_file = os.path.join(current_path, code_line_file)
    start_find(log_name_file, code_line_file, kernel_meta_file)

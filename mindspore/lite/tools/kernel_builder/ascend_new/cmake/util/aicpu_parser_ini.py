# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
aicpu ini parser
"""

import json
import os
import stat
import sys


OP_INFO_ENGINE_VALUE = ["DNN_VM_AICPU", "DNN_VM_HOST_CPU"]
OP_INFO_FLAGPARTIAL_VALUE = ["False"]
OP_INFO_COMPUTECOST_VALUE = ["100"]
OP_INFO_SUBTYPE_OF_INFERSHAPE_VALUE = ["1", "2", "3", "4"]
BOOL_LIST = ["True", "False"]
TYPE_LIST = ["DT_UINT8", "DT_UINT16", "DT_INT8", "DT_INT16", "DT_INT32", "DT_INT64", "DT_UINT32", "DT_UINT64",
             "DT_FLOAT16", "DT_FLOAT", "DT_DOUBLE", "DT_BOOL", "DT_COMPLEX64", "DT_COMPLEX128"]
FORMAT_LIST = ["ND", "NHWC", "NCHW"]


def parse_ini_files(ini_files):
    '''
    init all ini files
    '''
    aicpu_ops_info = {}
    for ini_file in ini_files:
        check_file_size(ini_file)
        parse_ini_to_obj(ini_file, aicpu_ops_info)
    return aicpu_ops_info


def check_file_size(input_file):
    try:
        file_size = os.path.getsize(input_file)
    except OSError as os_error:
        print('[ERROR] Failed to open "%s". %s' % (input_file, str(os_error)))
        raise OSError from os_error
    if file_size > 10*1024*1024:
        print('[WARN] The size of %s exceeds 10MB, it may take more time to run, please wait.' % input_file)


def parse_ini_to_obj(ini_file, aicpu_ops_info):
    '''
    parse all ini files to object
    '''
    with open(ini_file) as ini_read_file:
        lines = ini_read_file.readlines()
        ops = {}
        find_op_type = False
        for line in lines:
            line = line.rstrip()
            if line.startswith("["):
                if line.endswith("]"):
                    op_name = line[1:-1]
                    ops = {}
                    aicpu_ops_info[op_name] = ops
                    find_op_type = True
            elif "=" in line:
                key1 = line[:line.index("=")]
                key2 = line[line.index("=")+1:]
                key1_0, key1_1 = key1.split(".")
                if key1_0 not in ops:
                    ops[key1_0] = {}
                dic_key = ops.get(key1_0)
                dic_key[key1_1] = key2
            else:
                continue
        if not find_op_type:
            raise RuntimeError("Not find OpType in .ini file.")


def check_custom_op_opinfo(required_custom_op_info_keys, ops, op_key):
    '''
    check custom op info
    '''
    op_info = ops["opInfo"]
    missing_keys = []
    for required_op_info_key in required_custom_op_info_keys:
        if required_op_info_key not in op_info:
            missing_keys.append(required_op_info_key)
    if missing_keys:
        print("op: " + op_key + " opInfo missing: " + ",".join(missing_keys))
        raise KeyError("Check opInfo failed!")


def check_opinfo_value(op_info):
    """
    Function Description:
        Check opinfo value
    Parameter: op_info
    """
    is_valid = True
    check_key = ["engine", "flagPartial", "computeCost", "flagAsync", "subTypeOfInferShape", "flagSupportBlockDim"]
    key_support = [OP_INFO_ENGINE_VALUE, OP_INFO_FLAGPARTIAL_VALUE, OP_INFO_COMPUTECOST_VALUE, BOOL_LIST,
                   OP_INFO_SUBTYPE_OF_INFERSHAPE_VALUE, BOOL_LIST]
    for key, value_list in zip(check_key, key_support):
        info_value = op_info.get(key)
        if info_value is not None and info_value.strip() not in value_list:
            is_valid = False
            print("opInfo.{0} only support {1}.".format(key, value_list))
    return is_valid


def check_op_opinfo(required_op_info_keys, required_custom_op_info_keys,
                    ops, op_key):
    '''
    check normal op info
    '''
    op_info = ops["opInfo"]
    missing_keys = []
    for required_op_info_key in required_op_info_keys:
        if required_op_info_key not in op_info:
            missing_keys.append(required_op_info_key)
    if missing_keys:
        print("op: " + op_key + " opInfo missing: " + ",".join(missing_keys))
        raise KeyError("Check opInfo required key failed.")
    if op_info["opKernelLib"] == "CUSTAICPUKernel":
        check_custom_op_opinfo(required_custom_op_info_keys, ops, op_key)
        ops["opInfo"]["userDefined"] = "True"
    if not check_opinfo_value(op_info):
        raise KeyError("Check opInfo value failed.")


def check_op_input_output(info, key, ops):
    '''
    check input and output infos of all ops
    '''
    op_input_output = ops.get(key)
    for op_sets in op_input_output:
        if op_sets not in ('format', 'type', 'name'):
            print(info + " should has format type or name as the key, "
                  + "but getting " + op_sets)
            raise KeyError("Check input and output info failed.")
        if not check_type_format(op_sets, op_input_output, key):
            raise KeyError("Check input and output type or format failed.")


def check_type_format(op_key, op_dict, key_input_output):
    """
    Function Description:
        Check type and format
    Parameter: op_key, such as name,type,format
    Parameter: op_dict
    Parameter: key_input_output, such as input0
    """
    is_valid = True
    type_format = ["type", "format"]
    type_format_value = [TYPE_LIST, FORMAT_LIST]
    for key, value_list in zip(type_format, type_format_value):
        if op_key == key:
            op_value_str = op_dict.get(key)
            is_valid = check_value_valid(is_valid, key_input_output, op_key, op_value_str, value_list)
    return is_valid


def check_value_valid(is_valid, key_input_output, op_key, op_value_str, value_list):
    """
    Function Description:
        Check value valid
    Parameter: is_valid, bool
    Parameter: key_input_output, such as input0
    Parameter: op_key, such as name,type,format
    Parameter: op_value_str,such as int8,int16
    Parameter: value_list, support value
    """
    if op_value_str:
        for op_value in op_value_str.split(","):
            if op_value.strip() not in value_list:
                is_valid = False
                print("{0}.{1} not support {2}.".format(key_input_output, op_key, op_value))
    return is_valid


def check_op_info(aicpu_ops):
    '''
    check all ops
    '''
    print("==============check valid for aicpu ops info start==============")
    required_op_info_keys = ["computeCost", "engine", "flagAsync",
                             "flagPartial", "opKernelLib", "kernelSo", "functionName"]
    required_custom_op_info_keys = ["workspaceSize"]

    for op_key in aicpu_ops:
        ops = aicpu_ops[op_key]
        for key in ops:
            if key == "opInfo":
                check_op_opinfo(required_op_info_keys,
                                required_custom_op_info_keys, ops, op_key)

            elif (key[:5] == "input") and (key[5:].isdigit()):
                check_op_input_output("input", key, ops)
            elif (key[:6] == "output") and (key[6:].isdigit()):
                check_op_input_output("output", key, ops)
            elif (key[:13] == "dynamic_input") and (key[13:].isdigit()):
                check_op_input_output("dynamic_input", key, ops)
            elif (key[:14] == "dynamic_output") and (key[14:].isdigit()):
                check_op_input_output("dynamic_output", key, ops)
            else:
                print("Only opInfo, input[0-9], output[0-9] can be used as a "
                      "key, but op %s has the key %s" % (op_key, key))
                raise KeyError("bad key value")
    print("==============check valid for aicpu ops info end================\n")


def write_json_file(aicpu_ops_info, json_file_path):
    '''
    write json file from ini file
    '''
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    modes = stat.S_IWUSR | stat.S_IRUSR
    json_file_real_path = os.path.realpath(json_file_path)
    with os.fdopen(os.open(json_file_real_path, flags, modes), "w") as json_file:
        # Only the owner and group have rights
        os.chmod(json_file_real_path, stat.S_IWGRP + stat.S_IWUSR + stat.S_IRGRP
                 + stat.S_IRUSR)
        json.dump(aicpu_ops_info, json_file, sort_keys=True,
                  indent=4, separators=(',', ':'))
    print("Compile aicpu op info cfg successfully.")


def dump_json(aicpu_ops_info, outfile_path_arg):
    '''
    dump_json
    '''
    check_op_info(aicpu_ops_info)
    write_json_file(aicpu_ops_info, outfile_path_arg)


def parse_ini_to_json(ini_file_paths_arg, outfile_path_arg):
    '''
    parse ini to json
    '''
    aicpu_ops_info = parse_ini_files(ini_file_paths_arg)
    check_flag = True
    try:
        dump_json(aicpu_ops_info, outfile_path_arg)
    except KeyError:
        check_flag = False
        print("bad format key value, failed to generate json file")
    return check_flag


if __name__ == '__main__':
    get_args = sys.argv

    OUTPUT = "tf_kernel.json"
    ini_file_paths = []

    for arg in get_args:
        if arg.endswith("ini"):
            ini_file_paths.append(arg)
        if arg.endswith("json"):
            OUTPUT = arg

    if not ini_file_paths:
        ini_file_paths.append("tf_kernel.ini")

    if not parse_ini_to_json(ini_file_paths, OUTPUT):
        sys.exit(1)
    sys.exit(0)

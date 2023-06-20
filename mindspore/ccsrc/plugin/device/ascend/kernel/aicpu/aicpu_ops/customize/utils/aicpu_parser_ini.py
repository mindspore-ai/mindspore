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
"""
aicpu ini parser
"""
import json
import os
import stat
import sys

cust_op_lists = [
    "Cast",
    "Conj",
    "IsNan",
    "SliceGrad",
    "MaskedSelectGrad",
    "GatherDGradV2"
]


def parse_ini_files(ini_files):
    '''
    init all ini files
    '''
    aicpu_ops_info = {}
    for ini_file in ini_files:
        parse_ini_to_obj(ini_file, aicpu_ops_info)
    return aicpu_ops_info


def parse_ini_to_obj(ini_file, aicpu_ops_info):
    '''
    parse all ini files to object
    '''
    with open(ini_file) as ini_read_file:
        lines = ini_read_file.readlines()
        op_name, info = None, {}
        for line in lines:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith("["):
                if op_name and info: # set info for the last op
                    aicpu_ops_info["Cust"+op_name] = info
                info = {}
                op_name = line[1:-1]
                info = {}
                if op_name not in cust_op_lists:
                    op_name = None
                    continue
                aicpu_ops_info[op_name] = info
            elif op_name:
                key1 = line[:line.index("=")].strip()
                key2 = line[line.index("=")+1:].strip()
                key1_0, key1_1 = key1.split(".")
                if key1_0 not in info:
                    info[key1_0] = {}
                info[key1_0][key1_1] = key2
        if op_name and info:
            aicpu_ops_info["Cust"+op_name] = info


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
        raise KeyError("bad key value")


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
        raise KeyError("bad key value")
    if op_info["opKernelLib"] == "CUSTAICPUKernel":
        check_custom_op_opinfo(required_custom_op_info_keys, ops, op_key)
        ops["opInfo"]["userDefined"] = "True"


def check_op_input_output(info, key, ops):
    '''
    check input and output infos of all ops
    '''
    for op_sets in ops[key]:
        if op_sets not in ('format', 'type', 'name'):
            print(info + " should has format type or name as the key, "
                  + "but getting " + op_sets)
            raise KeyError("bad op_sets key")


def check_op_info(aicpu_ops):
    '''
    check all ops
    '''
    print("==============check valid for aicpu ops info start==============")
    required_op_info_keys = ["computeCost", "engine", "flagAsync",
                             "flagPartial", "opKernelLib"]
    required_custom_op_info_keys = ["kernelSo", "functionName"]

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
    print("==============check valid for aicpu ops info end================")


def write_json_file(aicpu_ops_info, json_file_path):
    '''
    write json file from ini file
    '''
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    modes = stat.S_IWUSR | stat.S_IRUSR
    json_file_real_path = os.path.realpath(json_file_path)
    if os.path.exists(json_file_real_path):
        os.remove(json_file_real_path)
    with os.fdopen(os.open(json_file_real_path, flags, modes), "w") as json_file:
        # Only the owner and group have rights
        os.chmod(json_file_real_path, stat.S_IWGRP + stat.S_IWUSR + stat.S_IRGRP
                 + stat.S_IRUSR)
        json.dump(aicpu_ops_info, json_file, sort_keys=True,
                  indent=4, separators=(',', ':'))
    print("Compile aicpu op info cfg successfully.")


def parse_ini_to_json(ini_file_paths_arg, outfile_path_arg):
    '''
    parse ini to json
    '''
    aicpu_ops_info = parse_ini_files(ini_file_paths_arg)
    try:
        check_op_info(aicpu_ops_info)
    except KeyError:
        print("bad format key value, failed to generate json file")
    else:
        write_json_file(aicpu_ops_info, outfile_path_arg)


if __name__ == '__main__':
    get_args = sys.argv

    OUTPUT = "aicpu_kernel.json"
    ini_file_paths = []

    for arg in get_args:
        if arg.endswith("ini"):
            ini_file_paths.append(arg)
        if arg.endswith("json"):
            OUTPUT = arg

    if not ini_file_paths:
        ini_file_paths.append("aicpu_kernel.ini")

    parse_ini_to_json(ini_file_paths, OUTPUT)

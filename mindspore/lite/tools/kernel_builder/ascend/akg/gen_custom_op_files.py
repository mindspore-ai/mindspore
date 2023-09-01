#!/usr/bin/env python3
# coding: utf-8
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
"""Used to generate custom akg op files, which will be invoked by GE."""

import os
import sys
import json
import shutil
from itertools import product

SUPPORTED_INPUT_NUM = [1, 2, 3, 4, 5, 6, 7]
SUPPORTED_OUTPUT_NUM = [1, 2, 3, 4, 5]
SUPPORTED_DEVICE_ARCH = ["ascend910", "ascend310p", "ascend310", "ascend910b"]


def gen_ops_info():
    """Generate the custom akg op registration information."""
    ops_info = {}
    # supported_io_num: [(1, 1), (1, 2), ...], list of (input_num, output_num)
    supported_io_num = list(product(SUPPORTED_INPUT_NUM, SUPPORTED_OUTPUT_NUM))
    for input_num, output_num in supported_io_num:
        op_info = {"attr": {"list": "info_path"},
                   "attr_info_path": {"paramType": "required", "type": "str", "value": "all"},
                   "opFile": {"value": "custom"},
                   "opInterface": {"value": "custom"},
                   "dynamicFormat": {"flag": "true"}}
        for i in range(input_num):
            op_info["input" + str(i)] = {"name": "x" + str(i),
                                         "paramType": "required",
                                         "shape": "all"}
        for i in range(output_num):
            op_info["output" + str(i)] = {"name": "y" + str(i),
                                          "paramType": "required",
                                          "shape": "all"}
        op_type = "Fused_x{}_y{}".format(input_num, output_num)
        ops_info[op_type] = op_info
    return ops_info


def gen_custom_op_files(config_dir, dsl_dir):
    """Copy custom akg op registration information file to config_dir, and copy python dsl file to dsl_dir."""
    config_dir = os.path.realpath(config_dir)
    dsl_dir = os.path.realpath(dsl_dir)
    cur_path = os.path.split(os.path.realpath(__file__))[0]

    # generate custom akg op registration information
    ops_info = gen_ops_info()
    for device_arch in SUPPORTED_DEVICE_ARCH:
        sub_dir = os.path.join(config_dir, device_arch)
        if not os.path.isdir(sub_dir):
            os.makedirs(sub_dir)
        ops_info_path = os.path.join(sub_dir, "aic-{}-ops-info.json".format(device_arch))
        if os.path.isfile(ops_info_path):
            with open(ops_info_path, 'r') as f:
                info_des = json.loads(f.read())
                ops_info.update(info_des)
        with os.fdopen(os.open(ops_info_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o660), 'w') as f:
            f.write(json.dumps(ops_info, indent=4))

    # custom akg op dsl file
    custom_dsl = os.path.join(cur_path, "custom.py")
    compiler = os.path.join(cur_path, "compiler.py")
    shutil.copy(custom_dsl, dsl_dir)
    shutil.copy(compiler, dsl_dir)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        gen_custom_op_files(sys.argv[1], sys.argv[2])

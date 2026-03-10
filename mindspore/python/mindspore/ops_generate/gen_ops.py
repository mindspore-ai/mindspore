# Copyright 2023-2025 Huawei Technologies Co., Ltd
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
"""
Auto generate ops files.
"""
import logging

from resources.resource_manager import prepare_resources
from common import gen_utils

from op_def.gen_op_def import generate_ops_def_files
from op_def_py.gen_op_def_py import generate_ops_py_files
from api.gen_api import generate_api_files
from aclnn.aclnn_kernel_register_auto_cc_generator import generate_aclnn_reg_file
from pyboost.gen_pyboost_func import gen_pyboost_code


module_generators = [
    generate_ops_py_files,    # generate ops python files
    generate_ops_def_files,   # generate ops definition files
    gen_pyboost_code,         # generate pyboost code
    generate_aclnn_reg_file,  # generate aclnn kernelmod register
    generate_api_files        # generate api definition files
]


def main():
    resource_mgr = prepare_resources()

    for generator in module_generators:
        generator(resource_mgr)

    gen_utils.clear_obsolete_auto_gen_files()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical("Auto generate failed, err info: %s", e)
        raise e

# Copyright 2025 Huawei Technologies Co., Ltd
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
import os
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1", card_mark="onecard")
def test_support_binary_in_graph_mode():
    """
    Feature: Support binary in graph mode.
    Description: Support run pyc or so in graph mode.
    Expectation: Execute training successful.
    """
    file_path = os.getenv('HOME') + '/test_support_binary_include_so_and_pyc'
    return_code = os.system(
        f"export MS_SUPPORT_BINARY=1;"
        f"rm -rf {file_path};"
        f"mkdir {file_path};"
        f"cp -f lenet_train.py {file_path}; "
        f"cp -f run_lenet_train.py {file_path}; "
        f"python {file_path}/run_lenet_train.py"
    )
    assert return_code == 0

    file_name = f"{file_path}/lenet_train.py"
    with open(file_name, 'r') as f:
        lines = f.readlines()
        if all("setattr(LeNet.construct, 'source', ([" not in line for line in lines):
            raise ValueError("Add setattr for LeNet failed！")
    os.system("rm -rf ${HOME}/test_support_binary_include_so_and_pyc")

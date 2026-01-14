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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_dynamic_shape_execution_with_conf_thread_num():
    """
    Feature: Test dynamic shape execution with configure thread number.
    Description: Test dynamic shape execution with configure thread number 1,2 or 4, if thread_num <= 3,
                 disable runtime multi-pipeline, if thread_num == 1, disable async launch.
    Expectation: The program execute and exit normally.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(current_dir, "test_dynamic_shape_execution.py")
    command = (f"python {script} --thread_num 1")
    return_code = os.system(command)
    assert return_code == 0

    command = (f"python {script} --thread_num 2")
    return_code = os.system(command)
    assert return_code == 0

    command = (f"python {script} --thread_num 4")
    return_code = os.system(command)
    assert return_code == 0


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=0.01)
def test_computed_depend_case_with_conf_thread_num():
    """
    Feature: Test dynamic shape with computed depend ops(unique) execution and config thread number.
    Description: Test dynamic shape execution with configure thread number2,4, if thread_num <= 3,
                 disable runtime multi-pipeline, in this case, dynamic shape will shape some framework
                 process with static shape, if thread_num == 1, disable async launch. Framework wait
                for unique op to finish launch and update output shape.
    Expectation: The program execute and exit normally.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(current_dir, "test_dynamic_shape_computed_depend.py")

    parameters = [(2, True, True), (2, True, False), (2, False, False), (4, True, True)]
    for thread_num, enable_pipeline, enable_new_pipeline in parameters:
        command = (f"python {script} --thread_num {thread_num} --enable_pipeline {enable_pipeline} \
                   --enable_new_pipeline {enable_new_pipeline}")
        return_code = os.system(command)
        assert return_code == 0

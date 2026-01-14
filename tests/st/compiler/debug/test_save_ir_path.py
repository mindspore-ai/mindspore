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
import shutil
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="unessential")
def test_dump_ir_path_cpu():
    '''
    Feature: test ir save path.
    Description: Test ir save path of different rank using msrun.
    Expectation: Generate the expected ir path.
    '''
    case_name = "test_dump_ir_path_cpu"
    return_code = os.system(
        f"mkdir {case_name}; cd {case_name};"
        "msrun --worker_num 8 --local_worker_num 8 --join=True "
        "pytest -s ../save_ir_path_code.py::test_ir_path_with_distributed_initialized"
    )

    assert return_code == 0
    for i in range(8):
        assert os.path.exists(f"{case_name}/ir/rank_{i}")

    if os.path.exists(f"{case_name}/"):
        shutil.rmtree(f"{case_name}/")

@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential",
        )
def test_dump_ir_path_ascend():
    '''
    Feature: test ir save path.
    Description: Test ir save path of different rank using msrun.
    Expectation: Generate the expected ir path.
    '''
    case_name = "test_dump_ir_path_ascend"
    return_code = os.system(
        f"mkdir {case_name}; cd {case_name};"
        "msrun --worker_num 8 --local_worker_num 8 --join=True "
        "pytest -s ../save_ir_path_code.py::test_ir_path_with_distributed_initialized"
    )

    assert return_code == 0
    for i in range(8):
        assert os.path.exists(f"{case_name}/ir/rank_{i}")

    if os.path.exists(f"{case_name}/"):
        shutil.rmtree(f"{case_name}/")

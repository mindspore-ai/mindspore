# Copyright 2024 Huawei Technologies Co., Ltd
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

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards')
def test_matmul_allreduce_addrmsnorm_forward():
    """
    Feature: Test MatmulAllReduceAddRmsNorm forward.
    Description: Test in kbk and pynative mode with dtype float16 and bfloat16
    Expectation: Run success
    """
    os.environ["HCCL_DETERMINISTIC"] = "true"

    ret = os.system("msrun --worker_num=2 --local_worker_num=2 --join=True --master_port=8221 "
                    "pytest -s --disable-warnings "
                    "matmul_allreduce_add_rmsnorm.py::test_matmul_allreduce_addrmsnorm_forward")
    assert ret == 0


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards')
def test_matmul_allreduce_addrmsnorm_forward_dynamic_shape():
    """
    Feature: Test MatmulAllReduceAddRmsNorm forward with dynamic shape input
    Description: Test in kbk and pynative mode with dtype float16 and bfloat16
    Expectation: Run success
    """
    os.environ["HCCL_DETERMINISTIC"] = "true"

    ret = os.system("msrun --worker_num=2 --local_worker_num=2 --join=True --master_port=8222 "
                    "pytest -s --disable-warnings "
                    "matmul_allreduce_add_rmsnorm.py::test_matmul_allreduce_addrmsnorm_forward_dynamic_shape")
    assert ret == 0


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards')
def test_matmul_allreduce_addrmsnorm_forward_fusion():
    """
    Feature: Test MatmulAllReduceAddRmsNorm forward ir fusion pass
    Description: Test in kbk mode
    Expectation: Run success
    """
    os.environ["MS_ENABLE_INTERNAL_KERNELS"] = "on"
    os.environ["MS_ENABLE_LCCL"] = "off"
    os.environ["HCCL_DETERMINISTIC"] = "true"

    ret = os.system("msrun --worker_num=2 --local_worker_num=2 --join=True --master_port=8223 "
                    "pytest -s --disable-warnings "
                    "matmul_allreduce_add_rmsnorm.py::test_matmul_allreduce_addrmsnorm_forward_fusion")
    assert ret == 0

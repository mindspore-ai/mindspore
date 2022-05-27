# Copyright 2022 Huawei Technologies Co., Ltd
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
import pytest


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
def test_resnet_imagenet_8p_mpi():
    """
    Feature: Resnet50 network.
    Description: Train and evaluate resnet50 network on imagenet dataset with mpi.
    Expectation: accuracy > 0.1, time cost < 26.
    """
    os.environ['HCCL_WHITELIST_DISABLE'] = str(1)
    return_code = os.system("mpirun -n 8 pytest -s test_resnet50_imagenet.py::test_resnet_imagenet_8p_mpi")
    assert return_code == 0

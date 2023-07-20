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


def test_pynative_resnet50_gpu_8p_performance():
    """
    Feature: PyNative ResNet50 8P
    Description: test PyNative ResNet50 8p performance
    Expectation: success, return_code==0
    """
    return_code = os.system("mpirun -n 8 pytest -s test_pynative_resnet50_gpu.py")
    assert return_code == 0

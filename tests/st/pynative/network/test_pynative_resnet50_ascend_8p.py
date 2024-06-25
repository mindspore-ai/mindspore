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
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='allcards',
          essential_mark='essential')
def test_pynative_resnet50_ascend_8p_mpi():
    """
    Feature: PyNative ResNet50 8P
    Description: test PyNative ResNet50 8p with mpirun
    Expectation: success, return_code==0
    """
    os.system("mpirun -n 8 pytest -s test_pynative_resnet50_ascend.py::test_train_tensor"
              " >stdout.log 2>&1")
    return_code = os.system(r"grep '1 passed' stdout.log")
    assert return_code == 0

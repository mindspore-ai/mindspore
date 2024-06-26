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
import pytest
from mindspore import context
import shutil
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='allcards', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_parameter_broadcast(mode):
    """
    Feature: parameter_broadcast
    Description: test parameter broadcast
    Expectation: success
    """
    for i in range(8):
        os.mkdir("device" + str(i))
    cmd = "mpirun -n 8 --allow-run-as-root pytest -s -v op_parallel.py::test_param_broadcast"
    ret = os.system(cmd)
    assert ret == 0
    # Clean files
    for i in range(8):
        shutil.rmtree("device" + str(i))

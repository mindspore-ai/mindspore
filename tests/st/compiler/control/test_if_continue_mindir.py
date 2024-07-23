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
# ============================================================================
"""Export if continue mindir."""
import numpy as np
import os
from tests.st.compiler.control.cases_register import case_register
import mindspore as ms
from mindspore import Tensor, nn, export, load

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_context(device_target='Ascend')

class Conv_for_if_continue_net(nn.Cell):
    def __init__(self):
        super(Conv_for_if_continue_net, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, weight_init="ones")

    def construct(self, x, y):
        for _ in range(2):
            if y < 2:
                continue
            else:
                x = self.conv1(x)
            #x = self.conv1(x)
        return x


@case_register.level0
@case_register.target_gpu
def test_if_continue_mindir():
    """
    Feature: export complex if-continue mindir and run.
    Description: test export complex if-continue mindir and run.
    Expectation: load and run minddir successfully and the result is correct.
    """
    file_name = "Conv2d_for_if_continue"
    file_name_mindir = "{}.mindir".format(file_name)

    net = Conv_for_if_continue_net()
    input_tensor = np.ones([1, 1, 3, 3]).astype(np.float32)
    input_weight = np.ones([1,], dtype=np.int32) + 1
    input_standard = np.array([[[[25, 35, 25],
                                 [35, 49, 35],
                                 [25, 35, 25]]]]).astype(np.float32)
    export(net, Tensor(input_tensor), Tensor(input_weight), file_name=file_name, file_format="MINDIR")
    assert os.path.exists(file_name_mindir)

    graph = load(file_name_mindir)
    net_mindir = nn.GraphCell(graph)
    success = True
    for _ in range(10):
        result_mindir = net_mindir(Tensor(input_tensor), Tensor(input_weight))
        if not np.allclose(result_mindir.asnumpy(), input_standard, 1.0e-5, 1.0e-5):
            success = False
    os.remove(file_name_mindir)
    assert success

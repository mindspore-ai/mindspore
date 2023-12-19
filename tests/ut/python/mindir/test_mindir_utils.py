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
import numpy as np
import mindspore as ms
from mindspore.nn import Cell
from mindspore import Tensor, export, Parameter, dtype, context


def test_load_and_save_mindir():
    """
    Feature: Test MindIR load_mindir and save_mindir
    Description: load mindir from a file or save mindir to a file.
    Expectation: No exception.
    """

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.w = Parameter(Tensor([-2], dtype.float32), name="weight")
            self.b = Parameter(Tensor([-5], dtype.float32), name="bias")

        def construct(self, x, y):
            if len(x.shape) == 1:
                return y
            while y >= x:
                if self.b <= x:
                    return y
                if self.w < x:
                    return x
                x += 1

            return x + y

    context.set_context(mode=context.GRAPH_MODE)
    x = np.array([3], np.float32)
    y = np.array([0], np.float32)
    net = Net()
    export(net, Tensor(x), Tensor(y), file_name="ctrl", file_format='MINDIR')
    md = ms.load_mindir("ctrl.mindir")
    md.user_info["version"] = "pangu v100"
    md.user_info["version"] = "pangu v200"
    ms.save_mindir(md, "ctrl.mindir")
    ms.save_mindir(md, "ctrl_test")
    ms.save_mindir(md, "test/ctrl_test")
    md_new = ms.load_mindir("ctrl.mindir")
    assert md_new.user_info["version"] == "pangu v200"

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
import glob
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from tests.mark_utils import arg_mark

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = P.Add()

    def construct(self, x_, y_):
        return self.add(x_, y_)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=1024.01)
def test_save_cce_graph():
    """
    Feature: Save cce file for Ascend ops
    Description: Test save cce file in GRAPH_MODE
    Expectation: there are cce files saved in kernel_meta/kernel_meta_*/kernel_meta
    """
    os.environ["MS_COMPILER_OP_LEVEL"] = "1"
    cur_path = os.path.split(os.path.realpath(__file__))[0]
    cce_path = os.path.join(cur_path, "kernel_meta")
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    add = Net()
    x = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
    y = np.array([[7, 8, 9], [10, 11, 12]]).astype(np.float32)
    add(Tensor(x), Tensor(y))
    cce_file = glob.glob(cce_path + "/kernel_meta_*/kernel_meta/*.cce")[0]
    assert cce_file
    del os.environ["MS_COMPILER_OP_LEVEL"]

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
import shutil
import platform
import numpy as np
from tests.mark_utils import arg_mark
import mindspore.context as context
from mindspore import Tensor
from mindspore.nn import Cell
import mindspore.ops.operations as P


class Net(Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = P.Add()
        self.mul = P.Mul()

    def construct(self, x0, x1):
        add_res = self.add(x1, x0)
        res = self.mul(add_res, x1)
        return res


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_for_llvm():
    """
    Feature: easy test case for graph_kernel in cpu check whether ci has llvm.
    Description: cpu test case, use graph_kernel execute ops.
    Expectation: if ci has not llvm assrt False else assert True
    """
    if platform.system() == "Linux":
        # run on cpu with gpu package
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU",
                            enable_graph_kernel=True, graph_kernel_flags="--dump_as_text")
        i0 = np.random.uniform(1, 2, [1, 1024]).astype(np.float32)
        i1 = np.random.uniform(1, 2, [1024, 1024]).astype(np.float32)
        net_obj = Net()
        output = net_obj(Tensor(i0), Tensor(i1)).asnumpy().copy()
        expect = (i0 + i1) * i1
        assert os.path.exists("./graph_kernel_dump")
        shutil.rmtree("./graph_kernel_dump")
        assert np.allclose(output, expect, rtol=1.e-4, atol=1.e-4, equal_nan=True)
    else:
        pass

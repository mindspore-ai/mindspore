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
# ==============================================================================
import numpy as np
from tests.mark_utils import arg_mark
import mindspore as ms
from mindspore import nn, Tensor, ops
from .util import Capture, capture


class PrtNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.prt = ops.Print()

    def construct(self, y=None):
        if y is not None:
            self.prt("y:", y)
        else:
            print("y is None", y)


class AddNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.prt_net = PrtNet()

    def construct(self, x, y):
        z = x + y
        self.prt_net(z)
        return z


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_adjust_print_for_ge():
    """
    Feature: Validate opt pass keep node scope and id.
    Description: Test opt pass AdjustPrintForGe.
    Expectation: No exception and node id after opt pass is as expected.
    """
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend')

    cap = Capture('print_insert_placeholder_for_tensor_name', 'Print')
    with capture(cap):
        net = AddNet()
        x = Tensor(np.ones([2, 2]).astype(np.float32))
        y = Tensor(np.ones([2]).astype(np.float32))
        expect = np.ones([2, 2]).astype(np.float32) * 2
        output = net(x, y)
        assert np.allclose(output.asnumpy(), expect, 1.0e-5, 1.0e-5)

    patterns = ['Default/prt_net-PrtNet/Print-op',
                'Default/prt_net-PrtNet/Depend-op']
    cap.check_output(patterns)

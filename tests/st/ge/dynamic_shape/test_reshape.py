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
import mindspore.common.dtype as mstype
from mindspore import Tensor, nn
from tests.mark_utils import arg_mark


class ReshapeNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.i = 0
        self.j = 1

    def construct(self, x):
        xshape = x.shape
        idx = xshape[self.i]
        idy = xshape[self.j]
        yshape = (idy, idx)
        y = x.reshape(yshape)
        return y


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_reshape():
    """
    Feature: convert ge graph
    Description: test ge dynamic unify mindir pass
    Expectation: success
    """

    x = Tensor(np.array([[1, 2, 3]]).astype(np.float32))
    dyn_x = Tensor(shape=(None, None), dtype=mstype.float32)
    net = ReshapeNet()
    net.set_inputs(dyn_x)
    y = net(x)
    assert np.allclose(y.asnumpy(), np.array([[1], [2], [3]]).astype(np.float32))

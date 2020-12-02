# Copyright 2020 Huawei Technologies Co., Ltd
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
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

class NetCholesky(nn.Cell):
    def __init__(self):
        super(NetCholesky, self).__init__()
        self.cholesky = P.Cholesky()

    def construct(self, x):
        return self.cholesky(x)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cholesky_fp32():
    cholesky = NetCholesky()
    x = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]]).astype(np.float32)
    output = cholesky(Tensor(x, dtype=mstype.float32))
    expect = np.linalg.cholesky(x)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect) < tol).all()

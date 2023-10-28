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
from mindspore import Tensor, ops, context
from mindspore import nn
from mindspore import context
from mindspore.ops.composite import GradOperation
from mindspore.ops.auto_generate import baddbmm


def test_baddbmm_ascend():
    context.set_context(device_target="Ascend")
    input = Tensor(np.ones([1, 3, 3]).astype(np.float32))
    batch1 = Tensor(np.ones([1, 3, 4]).astype(np.float32))
    batch2 = Tensor(np.ones([1, 4, 3]).astype(np.float32))
    output = baddbmm(input, batch1, batch2, 1, 1)
    output = baddbmm(output, batch1, batch2, 1, 1)
    assert (output.asnumpy() == np.ones([1, 3, 3]).astype(np.float32) * 9).all()


class Baddbmm(nn.Cell):
    def __init__(self):
        super(Baddbmm, self).__init__()

    def construct(self, input, batch1, batch2):
        return baddbmm(input, batch1, batch2, 1, 1)


def test_baddbmm_grad():
    input = Tensor(np.ones([1, 3, 3]).astype(np.float32))
    batch1 = Tensor(np.ones([1, 3, 4]).astype(np.float32))
    batch2 = Tensor(np.ones([1, 4, 3]).astype(np.float32))
    network = Baddbmm()
    grad_op = GradOperation(get_all=True)(network)
    grad = grad_op(input, batch1, batch2)
    assert (grad[0].asnumpy() == np.ones([1, 3, 3]).astype(np.float32)).all()
    assert (grad[1].asnumpy() == np.ones([1, 3, 4]).astype(np.float32) * 3).all()
    assert (grad[2].asnumpy() == np.ones([1, 4, 3]).astype(np.float32) * 3).all()

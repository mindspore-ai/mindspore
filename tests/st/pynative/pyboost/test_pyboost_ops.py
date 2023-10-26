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

def test_baddbmm_ascend():
    context.set_context(device_target="Ascend")
    baddbmm = ops.Baddbmm()

    input = Tensor(np.ones([1, 3, 3]).astype(np.float32))
    batch1 = Tensor(np.ones([1, 3, 4]).astype(np.float32))
    batch2 = Tensor(np.ones([1, 4, 3]).astype(np.float32))

    output = baddbmm(input, batch1, batch2, 1, 1)
    output = baddbmm(output, batch1, batch2, 1, 1)
    assert (output.asnumpy() == np.ones([1, 3, 3]).astype(np.float32) * 9).all()


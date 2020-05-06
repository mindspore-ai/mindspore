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
from mindspore.ops import operations as P
from mindspore.nn import Cell
from mindspore.common.tensor import Tensor
from mindspore.train.model import Model
from mindspore import log as logger
from mindspore import context
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

class Greater(Cell):
    def __init__(self):
        super(Greater, self).__init__()
        self.greater = P.Greater()

    def construct(self, inputa, inputb):
        return self.greater(inputa, inputb)

def me_greater(inputa, inputb):
    net = Greater()
    net.set_train()
    model = Model(net)

    out = model.predict(inputa, inputb)
    logger.info("Check input a: ")
    logger.info(inputa)
    logger.info("Check input b: ")
    logger.info(inputb)
    return out.asnumpy()

@pytest.mark.ssd_tbe
def test_greater_2d_scalar0():
    a = np.random.randint(-5, 5, [8, 32]).astype(np.int32)
    b = np.random.randint(-5, 5, [8, 32]).astype(np.int32)
    out_me = me_greater(Tensor(a), Tensor(b))
    logger.info("Check me result:")
    logger.info(out_me)
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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import log as logger
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.dropoutdomask = P.DropoutDoMask()

    def construct(self, x, mask, keep_prob):
        return self.dropoutdomask(x, mask, keep_prob)


def test_net():
    x = np.random.randn(2, 5, 8).astype(np.float32)
    mask = np.random.randn(16).astype(np.uint8)
    keep_prob = 1

    ddm = Net()
    output = ddm(Tensor(x), Tensor(mask), Tensor(keep_prob))
    logger.info("***********x*********")
    logger.info(x)
    logger.info("***********mask*********")
    logger.info(mask)
    logger.info("***********keep_prob*********")
    logger.info(keep_prob)

    logger.info("***********output y*********")
    logger.info(output.asnumpy())

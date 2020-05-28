# Copyright 2019 Huawei Technologies Co., Ltd
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

import numpy as np

from mindspore import Tensor
from mindspore import context
from mindspore.common import dtype as mstype
from mindspore.common.api import _executor
from mindspore.nn.loss.loss import SoftmaxCrossEntropyExpand


def test_SoftmaxCrossEntropy():
    net = SoftmaxCrossEntropyExpand(sparse=True)
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    logit = Tensor(np.ones([64, 512]), dtype=mstype.float32)
    label = Tensor(np.ones([64]), dtype=mstype.int32)
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    net.set_auto_parallel()
    _executor.compile(net, logit, label)

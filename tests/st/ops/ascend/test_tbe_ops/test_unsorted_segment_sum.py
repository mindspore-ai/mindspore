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
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype

context.set_context(device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.seg_sum = P.UnsortedSegmentSum()

    def construct(self, x, segment_ids, num_segments):
        return self.seg_sum(x, segment_ids, num_segments)


def me_un_seg_sum(input_, indices, num_segments):
    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    out = net(Tensor(input_), Tensor(indices), Tensor(num_segments, mstype.int32))
    return out.asnumpy()


def comapre_un_seg_sum(shape, indices, num_segments, dtype):
    input_ = np.random.randn(*shape).astype(dtype)
    indices_me = np.array(indices).astype(np.int32)
    out_me = me_un_seg_sum(input_, indices_me, num_segments)
    print("-------------ms------------------")
    print(out_me)


def test_net():
    np.random.seed(0)
    indices = np.random.randint(0, 1280, 1280)
    comapre_un_seg_sum([1280, 768], indices, 8192, np.float32)

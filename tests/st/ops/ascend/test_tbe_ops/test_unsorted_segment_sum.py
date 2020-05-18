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
from mindspore.train.model import Model

context.set_context(device_target="Ascend")


class Net(nn.Cell):
    def __init__(self, num_segments):
        super(Net, self).__init__()
        self.seg_sum = P.UnsortedSegmentSum()
        self.num_segments = num_segments

    def construct(self, x, segment_ids):
        return self.seg_sum(x, segment_ids, self.num_segments)


def me_un_seg_sum(input, indices, num_segments):
    context.set_context(mode=context.GRAPH_MODE)
    net = Net(num_segments)
    net.set_train()
    model = Model(net)
    out = model.predict(Tensor(input), Tensor(indices))
    return out.asnumpy()


def comapre_un_seg_sum(shape, indices, num_segments, dtype):
    input = np.random.randn(*shape).astype(dtype)
    indices_me = np.array(indices).astype(np.int32)
    out_me = me_un_seg_sum(input, indices_me, num_segments)
    print("-------------ms------------------")
    print(out_me)


def test_net():
    indices = np.random.randint(0, 1280, 1280)
    comapre_un_seg_sum([1280, 768], indices, 8192, np.float32)

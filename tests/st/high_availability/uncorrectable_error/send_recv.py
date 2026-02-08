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
# ============================================================================
"""send recv"""

import os
import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter
from mindspore._c_expression import send_recv
from mindspore.communication import init, get_rank


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        val = 2 if get_rank() == 0 else 3
        self.param = Parameter(Tensor(np.ones([2, 8], dtype=np.float32) * val), name='param')

    def construct(self, x):
        out = self.param + x
        self.param = out
        return out

init()
ms.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

if __name__ == '__main__':
    rank_size = os.environ['RANK_SIZE']

    print(f'rank_id={get_rank()}/{rank_size}')
    input_x = Tensor(np.arange(0, 16, dtype=np.float32).reshape(2, 8))
    net = Net()
    out = net(input_x)

    tag = 'send' if get_rank() == 0 else 'recv'
    print(f'param before {tag} {out.asnumpy()}', flush=True)

    send_recv([net.param], src_rank=0, dst_rank=1)
    print(f'param after {tag} {net.param.asnumpy()}', flush=True)

    assert(np.allclose(out.asnumpy(), net.param.asnumpy() + get_rank()))

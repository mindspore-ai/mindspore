# Copyright 2022 Huawei Technologies Co., Ltd
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
from mindspore.nn import Cell
from mindspore import context
from mindspore.ops. operations._ocr_ops import StringLength
from mindspore import Tensor
from mindspore import dtype as mstype
import numpy as np

context.set_context(mode=context.GRAPH_MODE)


class Net(Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.length = StringLength()

    def construct(self, x):
        return self.length(x)


def run_string_length():
    net = Net()
    a = np.array([["ab", "cde"], ["fghi", "jklmn"]])
    out = net(Tensor(a, dtype=mstype.string))
    expect_res = np.array([[2, 3], [4, 5]]).astype(np.int32)
    assert np.allclose(out.asnumpy(), expect_res)


if __name__ == "__main__":
    run_string_length()

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
import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore._checkparam import is_stub_tensor
from mindspore.ops.auto_generate.gen_ops_prim import select_ext_view_op, inplace_copy_op
from tests.mark_utils import arg_mark


context.set_context(mode=context.GRAPH_MODE)

@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_sync():
    """
    Feature: Tensor data sync.
    Description: Tensor need sync from device to host Sometimes.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self, x):
            super().__init__()
            y = x * 2
            self.z = y.stub_sync() if is_stub_tensor(y) else y
            self.w = x
        def construct(self, input1, input2):
            if self.z > self.w:
                return self.z - input1 - input2
            return self.z + input1 + input2

    input1 = ms.Tensor(5)
    input2 = ms.Tensor(3)
    ms.set_context(mode=ms.GRAPH_MODE)
    x = Tensor(1, dtype=ms.float16)
    net = Net(x)
    graph_out = net(input1, input2)
    print("graph_out:", graph_out)
    ms.set_context(mode=ms.PYNATIVE_MODE)
    pyantive_out = net(input1, input2)
    print("pyantive_out:", pyantive_out)
    assert (graph_out == pyantive_out).all()


@pytest.mark.skip(reason="has not supported")
@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_view_inplace_pynative():
    """
    Feature: View Inplace in PyNative mode.
    Description: View Inplace in PyNative mode.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self, x):
            super().__init__()
            self.x_viewed = select_ext_view_op(x, 0, 0)
            inplace_copy_op(self.x_viewed, ms.Tensor(-1))
            self.w = x
        def construct(self):
            return self.w


    x = ms.Tensor([1, 2, 3])
    net = Net(x)
    ms.set_context(mode=ms.GRAPH_MODE)
    graph_out = net()
    print("graph_out:", graph_out)
    except_out = [-1, 2, 3]
    ms.set_context(mode=ms.PYNATIVE_MODE)
    pyantive_out = net()
    print("pyantive_out:", pyantive_out)
    assert (graph_out == pyantive_out).all()
    assert (graph_out.asnumpy() == except_out).all()

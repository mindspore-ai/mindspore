# Copyright 2021 Huawei Technologies Co., Ltd
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
from mindspore.ops import composite as C
from mindspore.ops.operations import _inner_ops as inner

def test_gpu_convert_to_dynamic_shape_grad():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.op = inner.GpuConvertToDynamicShape()

        def construct(self, x1):
            return self.op(x1)


    class GradNet(nn.Cell):
        def __init__(self, network):
            super(GradNet, self).__init__()
            self.grad = C.GradOperation(get_all=True, sens_param=True)
            self.network = network

        def construct(self, x1, dy):
            return self.grad(self.network)(x1, dy)

    net = Net()
    grad_net = GradNet(net)

    x1 = Tensor(np.array([1.4, -1.2, 2.5, -3.23, -4.12, 5.53]).astype(np.float32))
    dy = Tensor(np.array([0.10, 0.11, 0.22, 0.33, 0.44, 0.155]).astype(np.float32))
    out = grad_net(x1, dy)
    np.testing.assert_allclose(out[0].asnumpy(), dy.asnumpy(), rtol=1e-6)

    x1 = Tensor(np.array([[4.4, -6.2], [22.5, 13.23], [293, 2.22]]).astype(np.float32))
    dy = Tensor(np.array([[0.001, 0.21], [0.22, 0.663], [0.422, 0.2]]).astype(np.float32))
    out = grad_net(x1, dy)
    np.testing.assert_allclose(out[0].asnumpy(), dy.asnumpy(), rtol=1e-6)

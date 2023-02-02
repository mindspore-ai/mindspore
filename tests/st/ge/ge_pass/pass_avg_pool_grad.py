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
""" test ge frontend pass `AvgPoolGradForGE` """
import numpy as np

from tests.st.ge import ge_infer_env  # pylint: disable=unused-import
import mindspore.context as context
import mindspore.nn as nn
from mindspore.ops import operations as op
from mindspore.common.tensor import Tensor
from mindspore.ops.composite import GradOperation


class AvgPoolNet(nn.Cell):
    def __init__(self, ksize=1, stride=1, padmode="VALID", data_format='NCHW'):
        super(AvgPoolNet, self).__init__()
        self.avgpool = op.AvgPool(kernel_size=ksize, strides=stride, pad_mode=padmode,
                                  data_format=data_format)

    def construct(self, input_x):
        return self.avgpool(input_x)


class GradNet(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.grad = GradOperation(sens_param=True)

    def construct(self, *inputs):
        grad_fn = self.grad(self.net)
        return grad_fn(*inputs)


def ge_avg_pool_grad_ksize2_stride2_same():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    net = AvgPoolNet(ksize=2, stride=2, padmode="SAME")
    grad_net = GradNet(net)
    x = Tensor(np.array([[[[10, 1, 2, 3, -4, -5],
                           [6, 7, 8, 9, -10, -11],
                           [12, 13, 24, -15, -16, -17],
                           [18, 19, 20, 21, 22, 23],
                           [32, 25, 26, 27, 28, 40],
                           [30, 31, 35, 33, 34, 35]]]]).astype(np.float32))
    sens = Tensor(np.arange(1, 10).reshape((1, 1, 3, 3)).astype(np.float32))
    out = grad_net(x, sens)
    return out


def run_ge_avg_pool_grad_ksize2_stride2_same():
    """
    Feature: Auto-diff `AvgPool` pipeline implement in ge
    Description: run the whole graph sink in ascend in ge backend
    Expectation: success
    """
    out = ge_avg_pool_grad_ksize2_stride2_same()
    expect_grad = np.array([[[[0.25, 0.25, 0.5, 0.5, 0.75, 0.75],
                              [0.25, 0.25, 0.5, 0.5, 0.75, 0.75],
                              [1., 1., 1.25, 1.25, 1.5, 1.5],
                              [1., 1., 1.25, 1.25, 1.5, 1.5],
                              [1.75, 1.75, 2., 2., 2.25, 2.25],
                              [1.75, 1.75, 2., 2., 2.25, 2.25]]]]).astype(np.float32)
    assert np.allclose(out.asnumpy(), expect_grad)


if __name__ == "__main__":
    run_ge_avg_pool_grad_ksize2_stride2_same()

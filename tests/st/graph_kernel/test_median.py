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

import numpy as np
from tests.mark_utils import arg_mark
import torch
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.ops.operations.math_ops import Median
import mindspore.ops.operations._grad_ops as G
from mindspore.ops.composite import GradOperation


class Grad(Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=False, sens_param=False)
        self.network = network

    def construct(self, input_x):
        gout = self.grad(self.network)(input_x)
        return gout


class MedianC(Cell):
    def __init__(self, global_median, axis, keep_dims):
        super().__init__()
        self.global_median = global_median
        self.axis = axis
        self.keep_dims = keep_dims
        self.median = Median(self.global_median, self.axis, self.keep_dims)

    def construct(self, x):
        return self.median(x)


class MedianGrad(Cell):
    def __init__(self, global_median, axis, keep_dims):
        super().__init__()
        self.global_median = global_median
        self.axis = axis
        self.keep_dims = keep_dims
        self.median_grad = G.MedianGrad(self.global_median, self.axis, self.keep_dims)

    def construct(self, dy, x, y, indices):
        return self.median_grad(dy, x, y, indices)


class MedianFactory():
    def __init__(self, input_shape, global_median, axis=0, keep_dims=False, dtype=np.float32):
        super().__init__()
        self.dtype = dtype
        self.input = np.random.randn(*input_shape).astype(self.dtype)
        self.global_median = global_median
        self.axis = axis
        self.keep_dims = keep_dims
        self.output_grad_np = np.random.randn(*input_shape).astype(dtype=dtype)

    def forward_mindspore_impl(self):
        net = MedianC(self.global_median, self.axis, self.keep_dims)
        y, indices = net(Tensor(self.input))
        return y.asnumpy(), indices.asnumpy()

    def grad_mindspore_impl(self):
        input_x = Tensor(self.input)
        net = MedianC(self.global_median, self.axis, self.keep_dims)
        grad_net = Grad(net)
        res = grad_net(input_x)
        return res.asnumpy()

    def forward_pytorch_impl(self):
        input_pt = torch.from_numpy(self.input)
        indices = None
        if self.global_median is False:
            y, indices = torch.median(input_pt, axis=self.axis, keepdim=self.keep_dims)
        else:
            y = torch.median(input_pt)
        indices_np = None if indices is None else indices.numpy().astype(np.int64)
        return y.numpy().astype(self.dtype), indices_np

    def global_grad_pytorch_impl(self):
        input_pt = torch.from_numpy(self.input)
        input_pt.requires_grad = True
        y = torch.median(input_pt)
        y.backward()
        return input_pt.grad.numpy()

    def grad_pytorch_impl(self):
        input_pt = torch.from_numpy(self.input)
        input_pt.requires_grad = True
        y, _ = torch.median(input_pt, axis=self.axis, keepdim=self.keep_dims)
        y.sum().backward()
        return input_pt.grad.numpy()

    def forward_cmp(self):
        y_pytorch, _ = self.forward_pytorch_impl()
        y_mindspore, _ = self.forward_mindspore_impl()
        assert np.allclose(y_pytorch, y_mindspore)

    def grad_cmp(self):
        grad_ms = self.grad_mindspore_impl()
        if self.global_median is False:
            grad_torch = self.grad_pytorch_impl()
        else:
            grad_torch = self.global_grad_pytorch_impl()
        assert np.allclose(grad_ms, grad_torch)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_median_gpu():
    """
    Feature: Test median.
    Description: Test median and mediangrad in Gpu with different global_median parameter.
    Expectation: the result match given one.
    """
    fact = MedianFactory(input_shape=(5, 5), global_median=True, axis=0, keep_dims=False)
    fact.forward_cmp()
    fact.grad_cmp()
    fact2 = MedianFactory(input_shape=(5, 5, 5), global_median=False, axis=1, keep_dims=True)
    fact2.forward_cmp()
    fact2.grad_cmp()

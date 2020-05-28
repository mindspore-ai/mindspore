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
import os
from numpy import allclose

import mindspore.communication.management as distributedTool
from mindspore import context
from mindspore._checkparam import check_bool, twice
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.ops.composite import grad_all_with_sens

device_num = 4
device_id = int(os.environ["RANK_ID"])
path = "./output/"


def setup_module():
    print("~~~~~~~~~~~set up~~~~~~~~~~~~~")
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(device_num=device_num, global_rank=device_id)
    distributedTool.init()
    distributedTool.create_group("0-3", [0, 1, 2, 3])
    print("~~~~~~~~~~~set up finished~~~~~~~~~~~~~")


def teardown_module():
    print("~~~~~~~~~~~~tear down~~~~~~~~~~")


class _Conv(Cell):
    r"""Applies a N-D convolution over an input signal composed of several input
       planes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 pad_mode,
                 padding,
                 dilation,
                 group,
                 has_bias,
                 weight_init,
                 bias_init):
        super(_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_mode = pad_mode
        self.padding = padding
        self.dilation = dilation
        self.group = group
        self.has_bias = has_bias
        if not (isinstance(in_channels, int) and in_channels > 0):
            raise ValueError('Attr \'in_channels\' of \'Conv2D\' Op passed '
                             + str(in_channels) + ', should be a int and greater than 0.')
        if (not isinstance(kernel_size, tuple)) or len(kernel_size) != 2 or \
                (not isinstance(kernel_size[0], int)) or (not isinstance(kernel_size[1], int)) or \
                kernel_size[0] < 1 or kernel_size[1] < 1:
            raise ValueError('Attr \'kernel_size\' of \'Conv2D\' Op passed '
                             + str(self.kernel_size) + ', should be a int or tuple and equal to or greater than 1.')
        if in_channels % group != 0:
            raise ValueError('Attr \'in_channels\' of \'Conv2D\' Op must be divisible by '
                             'attr \'group\' of \'Conv2D\' Op.')
        if out_channels % group != 0:
            raise ValueError('Attr \'out_channels\' of \'Conv2D\' Op must be divisible by '
                             'attr \'group\' of \'Conv2D\' Op.')

        self.weight = Parameter(initializer(
            weight_init, [out_channels, in_channels // group, *kernel_size]), name='weight')

        if check_bool(has_bias):
            self.bias = Parameter(initializer(
                bias_init, [out_channels]), name='bias')
        else:
            if bias_init != 'zeros':
                print("Value of 'has_bias' is False, value of 'bias_init' will be ignored.")
            self.bias = None

    def construct(self, *inputs):
        raise NotImplementedError


class Conv2d(_Conv):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 pad_mode='same',
                 padding=0,
                 dilation=1,
                 group=1,
                 has_bias=False,
                 weight_init='normal',
                 bias_init='zeros',
                 strategy=None):
        kernel_size = twice(kernel_size)
        super(Conv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            pad_mode,
            padding,
            dilation,
            group,
            has_bias,
            weight_init,
            bias_init)
        self.add = P.TensorAdd(strategy)
        self.conv2d = P.Conv2D(out_channel=self.out_channels,
                               kernel_size=self.kernel_size,
                               mode=1,
                               pad_mode=self.pad_mode,
                               pad=self.padding,
                               stride=self.stride,
                               dilation=self.dilation,
                               group=self.group,
                               strategy=None)
        self.bias_add = P.BiasAdd()

    def construct(self, input1, input2):
        x = self.add(input1, input2)
        if self.has_bias:
            return self.bias_add(self.conv2d(x, self.weight),
                                 self.bias)
        return self.conv2d(x, self.weight)


class Grad(Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.network = network

    def construct(self, input1, input2, output_grad):
        return grad_all_with_sens(self.network)(input1, input2, output_grad)


class Conv2dFactory:
    def __init__(self, input_shape, filter_shape, stride, pad_mode, padding, dilation, group, has_bias):
        self.in_n, self.in_c, self.in_h, self.in_w = input_shape
        self.out_c, self.kernel_c, self.kernel_h, self.kernel_w = filter_shape
        self.stride = stride
        self.pad_mode = pad_mode
        self.padding = padding
        self.dilation = dilation
        self.group = group
        self.strategy0 = (0, (4, 1, 1, 1), (1, 1, 1, 1))
        prefix = ""
        input_size = 1
        filter_size = 1
        for s in input_shape:
            prefix = prefix + str(s) + "_"
            input_size = input_size * s
        self.prefix = prefix
        for s in filter_shape:
            filter_size = filter_size * s
        number_range1 = min(10, input_size)
        number_range2 = min(10, filter_size)
        self.input_np1 = np.reshape(np.arange(0, input_size) % number_range1 - number_range1 / 2, input_shape).astype(
            np.float16)
        self.input_np2 = np.reshape(np.arange(0, input_size) % number_range1 - number_range1 / 4, input_shape).astype(
            np.float16)
        self.weight_np = np.reshape(np.arange(0, filter_size) % number_range2 - number_range2 / 2, filter_shape).astype(
            np.float16)
        self.has_bias = has_bias
        if self.has_bias is True:
            self.bias_np = np.arange(0, self.out_c).astype(np.float16)

        self.out_shape = (128, 64, 56, 56)
        out_size = 1
        for s in self.out_shape:
            out_size = out_size * s
        number_range3 = min(10, out_size)
        self.output_grad_np = np.reshape(np.arange(0, out_size) % number_range3 - number_range3 / 2,
                                         self.out_shape).astype(np.float16)
        self.x_id = device_id % 4
        self.y_id = device_id % 4
        self.out_strategy = self.strategy0[1]
        self.out_id = device_id % 4

    def get_parallel_blocks(self, input_, strategy):
        blocks = [input_]
        i = 0
        for stra in strategy:
            temp = []
            while len(blocks) > 0:
                block = blocks.pop(0)
                temp.extend(np.split(block, stra, axis=i))
            blocks.extend(temp)
            i += 1
        return blocks

    def forward_conv2d_mindspore_impl(self):
        input1 = Tensor(self.input_np1)
        input2 = Tensor(self.input_np2)
        weight = Tensor(self.weight_np)
        if self.has_bias:
            bias = Tensor(self.bias_np)
            net = Conv2d(in_channels=self.in_c, out_channels=self.out_c,
                         kernel_size=(self.kernel_h, self.kernel_w),
                         stride=self.stride, pad_mode=self.pad_mode,
                         padding=self.padding, dilation=self.dilation,
                         group=self.group, has_bias=True, weight_init=weight,
                         bias_init=bias)
        else:
            net = Conv2d(in_channels=self.in_c, out_channels=self.out_c,
                         kernel_size=(self.kernel_h, self.kernel_w),
                         stride=self.stride, pad_mode=self.pad_mode,
                         padding=self.padding, dilation=self.dilation,
                         group=self.group, has_bias=False, weight_init=weight)
        out = net(input1, input2)
        return out.asnumpy()

    def forward_conv2d_mindspore_parallel_impl(self):
        x = Tensor(self.input_np1)
        y = Tensor(self.input_np2)
        weight = Tensor(self.weight_np)
        inputs_x = self.get_parallel_blocks(self.input_np1, self.strategy0[1])
        inputs_y = self.get_parallel_blocks(self.input_np2, self.strategy0[1])
        x1 = Tensor(inputs_x[self.x_id])
        y1 = Tensor(inputs_y[self.y_id])
        if self.has_bias:
            bias = Tensor(self.bias_np)
            net = Conv2d(in_channels=self.in_c, out_channels=self.out_c,
                         kernel_size=(self.kernel_h, self.kernel_w),
                         stride=self.stride, pad_mode=self.pad_mode,
                         padding=self.padding, dilation=self.dilation,
                         group=self.group, has_bias=True, weight_init=weight,
                         bias_init=bias, strategy=(self.strategy0[0], self.strategy0[1], self.strategy0[1]))
        else:
            net = Conv2d(in_channels=self.in_c, out_channels=self.out_c,
                         kernel_size=(self.kernel_h, self.kernel_w),
                         stride=self.stride, pad_mode=self.pad_mode,
                         padding=self.padding, dilation=self.dilation,
                         group=self.group, has_bias=False, weight_init=weight,
                         strategy=(self.strategy0[0], self.strategy0[1], self.strategy0[1]))
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
        net.set_auto_parallel()
        out = net(x, y, parallel_inputs_compile=[x, y], parallel_inputs_run=[x1, y1])
        return out.asnumpy()

    def grad_conv2d_mindspore_impl(self):
        x = Tensor(self.input_np1)
        y = Tensor(self.input_np2)
        weight = Tensor(self.weight_np)
        output_grad = Tensor(self.output_grad_np)
        if self.has_bias:
            bias = Tensor(self.bias_np)
            net = Conv2d(in_channels=self.in_c, out_channels=self.out_c,
                         kernel_size=(self.kernel_h, self.kernel_w),
                         stride=self.stride, pad_mode=self.pad_mode,
                         padding=self.padding, dilation=self.dilation,
                         group=self.group, has_bias=True, weight_init=weight,
                         bias_init=bias, )
        else:
            net = Conv2d(in_channels=self.in_c, out_channels=self.out_c,
                         kernel_size=(self.kernel_h, self.kernel_w),
                         stride=self.stride, pad_mode=self.pad_mode,
                         padding=self.padding, dilation=self.dilation,
                         group=self.group, has_bias=False, weight_init=weight)

        grad_net = Grad(net)
        grad_net.set_train()
        out_grad = grad_net(x, y, output_grad)
        return out_grad

    def grad_conv2d_mindspore_parallel_impl(self):
        x = Tensor(self.input_np1)
        y = Tensor(self.input_np2)
        weight = Tensor(self.weight_np)
        inputs_x = self.get_parallel_blocks(self.input_np1, self.strategy0[1])
        inputs_y = self.get_parallel_blocks(self.input_np2, self.strategy0[1])
        x1 = Tensor(inputs_x[self.x_id])
        y1 = Tensor(inputs_y[self.y_id])
        output_grad = Tensor(self.output_grad_np)
        output_grads = self.get_parallel_blocks(self.output_grad_np, self.out_strategy)
        output_grad1 = Tensor(output_grads[self.out_id])
        if self.has_bias:
            bias = Tensor(self.bias_np)
            net = Conv2d(in_channels=self.in_c, out_channels=self.out_c,
                         kernel_size=(self.kernel_h, self.kernel_w),
                         stride=self.stride, pad_mode=self.pad_mode,
                         padding=self.padding, dilation=self.dilation,
                         group=self.group, has_bias=True, weight_init=weight,
                         bias_init=bias, strategy=(self.strategy0[0], self.strategy0[1], self.strategy0[1]))
        else:
            net = Conv2d(in_channels=self.in_c, out_channels=self.out_c,
                         kernel_size=(self.kernel_h, self.kernel_w),
                         stride=self.stride, pad_mode=self.pad_mode,
                         padding=self.padding, dilation=self.dilation,
                         group=self.group, has_bias=False, weight_init=weight,
                         strategy=(self.strategy0[0], self.strategy0[1], self.strategy0[1]))

        grad_net = Grad(net)
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
        grad_net.set_train()
        grad_net.set_auto_parallel()
        out_grad = grad_net(x, y, output_grad, parallel_inputs_compile=[x, y, output_grad1],
                            parallel_inputs_run=[x1, y1, output_grad1])
        return out_grad

    def forward_conv2d_cmp(self):
        out_mindspore = self.forward_conv2d_mindspore_impl()
        out_mindspore_parallel = self.forward_conv2d_mindspore_parallel_impl()
        out_blocks = self.get_parallel_blocks(out_mindspore, self.out_strategy)
        assert allclose(out_blocks[self.out_id], out_mindspore_parallel, 0.001, 0.001)

    def grad_conv2d_cmp(self):
        input_grad_mindspore = self.grad_conv2d_mindspore_impl()
        input_grad_mindspore_parallel = self.grad_conv2d_mindspore_parallel_impl()
        input_grad_mindspore0 = input_grad_mindspore[0].asnumpy()
        input_grad_mindspore1 = input_grad_mindspore[1].asnumpy()
        input_grad_mindspore_parallel0 = input_grad_mindspore_parallel[0].asnumpy()
        input_grad_mindspore_parallel1 = input_grad_mindspore_parallel[1].asnumpy()
        input_grad_blocks_0 = self.get_parallel_blocks(input_grad_mindspore0, self.strategy0[1])
        input_grad_blocks_1 = self.get_parallel_blocks(input_grad_mindspore1, self.strategy0[1])
        assert allclose(input_grad_blocks_0[self.x_id], input_grad_mindspore_parallel0, 0.001, 0.001)
        assert allclose(input_grad_blocks_1[self.x_id], input_grad_mindspore_parallel1, 0.001, 0.001)


def test_reid_conv2d_input_128_64_112_112_kernel_64_64_1_1_stride_2_padding_0_bias_true():
    fact = Conv2dFactory(input_shape=(128, 64, 112, 112),
                         filter_shape=(64, 64, 1, 1),
                         stride=2, pad_mode='valid', padding=0,
                         dilation=1, group=1, has_bias=False)
    fact.forward_conv2d_cmp()


def test_reid_conv2d_grad_input_128_64_112_112_kernel_64_64_1_1_stride_2_padding_0_bias_true():
    fact = Conv2dFactory(input_shape=(128, 64, 112, 112),
                         filter_shape=(64, 64, 1, 1),
                         stride=2, pad_mode='valid', padding=0,
                         dilation=1, group=1, has_bias=False)
    fact.grad_conv2d_cmp()

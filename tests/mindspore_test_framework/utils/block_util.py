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

"""Utils for Cell related computation."""

# pylint: disable=missing-docstring

import numpy as np

from mindspore import ParameterTuple
from mindspore import nn, context
from mindspore.common.api import _cell_graph_executor, ms_function
from mindspore.common.tensor import Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops.composite import GradOperation
from . import keyword


def get_uniform_with_shape(shape):
    np.random.seed(1)
    return np.random.uniform(-0.1, 0.1, size=shape).astype(np.float32)


def set_block_param_with_rand(net, rand_func=None):
    if not isinstance(net, nn.Cell) or rand_func is None:
        return
    net.init_parameters_data()
    for param in net.trainable_params():
        param.set_data(Tensor(rand_func(param.data.asnumpy().shape)))


def compile_block(net, *inputs, rand_func=None, training=True):
    set_block_training(net, training)
    set_block_param_with_rand(net, rand_func)
    return _cell_graph_executor.compile(net, *inputs)


def run_block(net, *inputs, rand_func=None, training=True):
    set_block_training(net, training)
    set_block_param_with_rand(net, rand_func)
    if context.get_context("mode") == context.PYNATIVE_MODE:
        def func_pynative(*inputs):
            @ms_function
            def _func_pynative(*inputs):
                return net(*inputs)

            return _func_pynative(*inputs)

        return func_pynative(*inputs)
    return net(*inputs)


class IthOutputCell(nn.Cell):
    def __init__(self, network, output_index):
        if isinstance(network, nn.Cell):
            super(IthOutputCell, self).__init__(auto_prefix=False)
        else:
            super(IthOutputCell, self).__init__()
        self.network = network
        self.output_index = output_index

    def construct(self, *inputs):
        predict = self.network(*inputs)[self.output_index]
        return predict


def get_output_cell(network, num_input, output_index, training=True):
    _ = num_input
    net = IthOutputCell(network, output_index)
    set_block_training(net, training)
    return net


class OutputReduceSumCell(nn.Cell):
    def __init__(self, network, output_num):
        super(OutputReduceSumCell, self).__init__()
        self.output_num = output_num
        self.network = network
        self.reduce_sum = P.ReduceSum()

    def construct(self, *inputs):
        if self.output_num == 1:
            return self.reduce_sum(self.network(*inputs), None)
        ret = F.make_tuple()
        for index in range(self.output_num):
            predict = self.network(*inputs)[index]
            predict_reduce = self.reduce_sum(predict, None)
            ret = ret + F.make_tuple(predict_reduce)
        return ret


def get_output_reduce_cell(network, output_num, training=True):
    net = OutputReduceSumCell(network, output_num)
    set_block_training(net, training)
    return net


class InputOpNet(nn.Cell):
    def __init__(self, op, c1=None, c2=None, c3=None, c4=None):
        super(InputOpNet, self).__init__()
        self.op = op
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4

    def construct(self, *inputs):
        raise NotImplementedError

    def construct0_c0_fake(self, data):
        x = self.op() + data
        return x

    def construct0_c1_fake(self, data):
        x = self.op(self.c1) + data
        return x

    def construct0_c2_fake(self, data):
        x = self.op(self.c1, self.c2) + data
        return x

    def construct0_c3_fake(self, data):
        x = self.op(self.c1, self.c2, self.c3) + data
        return x

    def construct0_c0(self):
        x = self.op()
        return x

    def construct0_c1(self):
        x = self.op(self.c1)
        return x

    def construct0_c2(self):
        x = self.op(self.c1, self.c2)
        return x

    def construct1_c0(self, x1):
        x = self.op(x1)
        return x

    def construct1_c1(self, x1):
        x = self.op(x1, self.c1)
        return x

    def construct1_c2(self, x1):
        x = self.op(x1, self.c1, self.c2)
        return x

    def construct1_c3(self, x1):
        x = self.op(x1, self.c1, self.c2, self.c3)
        return x

    def construct1_c4(self, x1):
        x = self.op(x1, self.c1, self.c2, self.c3, self.c4)
        return x

    def constructc1_1(self, x1):
        x = self.op(self.c1, x1)
        return x

    def construct2_c0(self, x1, x2):
        x = self.op(x1, x2)
        return x

    def construct2_c1(self, x1, x2):
        x = self.op(x1, x2, self.c1)
        return x

    def construct2_c3(self, x1, x2):
        x = self.op(x1, x2, self.c1, self.c2, self.c3)
        return x

    def construct3_c0(self, x1, x2, x3):
        x = self.op(x1, x2, x3)
        return x

    def construct3_c1(self, x1, x2, x3):
        x = self.op(x1, x2, x3, self.c1)
        return x

    def construct4_c0(self, x1, x2, x3, x4):
        x = self.op(x1, x2, x3, x4)
        return x

    def construct4_c1(self, x1, x2, x3, x4):
        x = self.op(x1, x2, x3, x4, self.c1)
        return x

    def construct4_c2(self, x1, x2, x3, x4):
        x = self.op(x1, x2, x3, x4, self.c1, self.c2)
        return x

    def construct4_c4(self, x1, x2, x3, x4):
        x = self.op(x1, x2, x3, x4, self.c1, self.c2, self.c3, self.c4)
        return x

    def construct5_c0(self, x1, x2, x3, x4, x5):
        x = self.op(x1, x2, x3, x4, x5)
        return x

    def construct6_c0(self, x1, x2, x3, x4, x5, x6):
        x = self.op(x1, x2, x3, x4, x5, x6)
        return x

    def construct5_c1(self, x1, x2, x3, x4, x5):
        x = self.op(x1, x2, x3, x4, x5, self.c1)
        return x

    def construct5_c4(self, x1, x2, x3, x4, x5):
        x = self.op(x1, x2, x3, x4, x5, self.c1, self.c2, self.c3, self.c4)
        return x

    def construct7_c0(self, x1, x2, x3, x4, x5, x6, x7):
        x = self.op(x1, x2, x3, x4, x5, x6, x7)
        return x

    def construct8_c0(self, x1, x2, x3, x4, x5, x6, x7, x8):
        x = self.op(x1, x2, x3, x4, x5, x6, x7, x8)
        return x

    def construct9_c0(self, x1, x2, x3, x4, x5, x6, x7, x8, x9):
        x = self.op(x1, x2, x3, x4, x5, x6, x7, x8, x9)
        return x


def gen_net(op, input_num, training=True, desc_const=(), const_first=False, add_fake_input=False):
    if isinstance(op, nn.Cell):
        return op
    net = InputOpNet(op, *desc_const)
    if const_first:
        fn_name = 'constructc%d_%d' % (len(desc_const), input_num)
    else:
        fn_name = 'construct%d_c%d' % (input_num, len(desc_const))
    if add_fake_input:
        fn_name += '_fake'
    f = getattr(net, fn_name)
    setattr(net, "construct", f)
    set_block_training(net, training)
    return net


class OperationBackward(nn.Cell):
    def __init__(self, network, grad_op, sens):
        if isinstance(network, nn.Cell):
            super(OperationBackward, self).__init__(auto_prefix=False)
        else:
            super(OperationBackward, self).__init__()
        self.network = network
        self.grad = grad_op
        self.sens = sens

    def construct(self, *inputs):
        return self.grad(self.network)(*inputs, self.sens)


class OperationBackwardWithNoSens(nn.Cell):
    def __init__(self, network, grad_op):
        if isinstance(network, nn.Cell):
            super(OperationBackwardWithNoSens, self).__init__(auto_prefix=False)
        else:
            super(OperationBackwardWithNoSens, self).__init__()
        self.network = network
        self.grad = grad_op

    def construct(self, *inputs):
        return self.grad(self.network)(*inputs)


class NNBackward(nn.Cell):
    def __init__(self, network, grad_op, sens):
        if isinstance(network, nn.Cell):
            super(NNBackward, self).__init__(auto_prefix=False)
        else:
            super(NNBackward, self).__init__()
        self.network = network
        self.grad = grad_op
        self.params = ParameterTuple(network.trainable_params())
        self.sens = sens

    def construct(self, *inputs):
        return self.grad(self.network, self.params)(*inputs, self.sens)


class NNBackwardWithNoSens(nn.Cell):
    def __init__(self, network, grad_op):
        if isinstance(network, nn.Cell):
            super(NNBackwardWithNoSens, self).__init__(auto_prefix=False)
        else:
            super(NNBackwardWithNoSens, self).__init__()
        self.network = network
        self.grad = grad_op
        self.params = ParameterTuple(network.trainable_params())

    def construct(self, *inputs):
        return self.grad(self.network, self.params)(*inputs)


def gen_grad_net(net, grad_op, input_num, sens=None, training=True, desc_const=(),
                 const_first=False, add_fake_input=False):
    if not isinstance(net, nn.Cell):
        net = gen_net(net, input_num, desc_const=desc_const, const_first=const_first, add_fake_input=add_fake_input)
    if grad_op.get_by_list:
        if grad_op.sens_param:
            net = NNBackward(net, grad_op, sens)
        else:
            net = NNBackwardWithNoSens(net, grad_op)
    else:
        if grad_op.sens_param:
            net = OperationBackward(net, grad_op, sens)
        else:
            net = OperationBackwardWithNoSens(net, grad_op)
    set_block_training(net, training)
    return net


def set_block_training(net, training=True):
    if isinstance(net, nn.Cell):
        net.set_train(training)


def set_block_phase(net, phase='train'):
    if isinstance(net, nn.Cell):
        net.phase = phase


def create_funcs(verification_set, block_generator, block_runner, grad_op=None, default_rand_func=None):
    def create_func(block, num_outputs, rand_func, desc_const, const_first, add_fake_input, split_outputs):
        def function(*inputs):
            # gradient
            if grad_op:
                if num_outputs == 0:
                    grad_op_ = GradOperation(get_all=grad_op.get_all,
                                             get_by_list=grad_op.get_by_list, sens_param=False)
                    b = block_generator(block, grad_op_, len(inputs), desc_const=desc_const,
                                        const_first=const_first, add_fake_input=add_fake_input)
                    return block_runner(b, *inputs, rand_func=rand_func)
                if num_outputs == 1:
                    b = block_generator(block, grad_op, len(inputs) - 1, inputs[-1], desc_const=desc_const,
                                        const_first=const_first, add_fake_input=add_fake_input)
                    return block_runner(b, *(inputs[:-1]), rand_func=rand_func)
                if split_outputs:
                    block_inputs = inputs[0:len(inputs) - num_outputs]
                    sens_inputs = inputs[len(inputs) - num_outputs:]
                    ret = []
                    for i in range(num_outputs):
                        bi_inputs = list(block_inputs)
                        bi = get_output_cell(block, len(block_inputs), i)
                        bi = block_generator(bi, grad_op, len(bi_inputs), sens_inputs[i], desc_const=desc_const,
                                             const_first=const_first, add_fake_input=add_fake_input)
                        grads_i = block_runner(bi, *bi_inputs, rand_func=rand_func)
                        if isinstance(grads_i, tuple):
                            ret.extend(grads_i)
                        else:
                            ret.append(grads_i)
                    return ret
                block_inputs = inputs[0:len(inputs) - num_outputs]
                sens_inputs = tuple(inputs[len(inputs) - num_outputs:])
                b = block_generator(block, grad_op, len(block_inputs), sens_inputs, desc_const=desc_const,
                                    const_first=const_first, add_fake_input=add_fake_input)
                return block_runner(b, *block_inputs, rand_func=rand_func)
            # forward
            inputs_num = len(inputs)
            if add_fake_input and inputs_num == 1:
                # input is faked
                inputs_num = 0
            b = block_generator(block, inputs_num, desc_const=desc_const, const_first=const_first,
                                add_fake_input=add_fake_input)
            return block_runner(b, *inputs, rand_func=rand_func)

        return function

    bc_configs = verification_set[keyword.function]
    for config in bc_configs:
        block = config[keyword.block]
        rand_func = config.get(keyword.init_param_with, default_rand_func)
        num_outputs = config.get(keyword.num_outputs, 0)
        desc_const = config.get(keyword.desc_const, [])
        const_first = config.get(keyword.const_first, False)
        add_fake_input = config.get(keyword.add_fake_input, False)
        split_outputs = config.get(keyword.split_outputs, True)
        config[keyword.block] = create_func(block, num_outputs, rand_func, desc_const,
                                            const_first, add_fake_input, split_outputs)
    return bc_configs

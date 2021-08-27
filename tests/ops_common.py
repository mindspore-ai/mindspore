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
""" test ops """
import numpy as np

import mindspore.nn as nn
import mindspore.ops.composite as C
import mindspore.ops.functional as F
import mindspore.ops.operations as P
from mindspore import Tensor
from mindspore.common.api import _cell_graph_executor


grad_all_with_sens = C.GradOperation(get_all=True, sens_param=True)


class InputBackward(nn.Cell):
    """ InputBackward definition """

    def __init__(self, network, c1=None, c2=None):
        super(InputBackward, self).__init__()
        self.network = network
        self.network.set_train()
        self.grad = grad_all_with_sens
        self.c1 = c1
        self.c2 = c2

    def construct(self, *inputs):
        pass

    def construct1(self, x1, sens):
        return self.grad(self.network)(x1, sens)

    def construct2(self, x1, x2, sens):
        return self.grad(self.network)(x1, x2, sens)

    def construct3(self, x1, x2, x3, sens):
        return self.grad(self.network)(x1, x2, x3, sens)

    def construct4(self, x1, x2, x3, x4, sens):
        return self.grad(self.network)(x1, x2, x3, x4, sens)

    def construct5(self, x1, x2, x3, x4, x5, sens):
        return self.grad(self.network)(x1, x2, x3, x4, x5, sens)

    def construct6(self, x1, x2, x3, x4, x5, x6, sens):
        return self.grad(self.network)(x1, x2, x3, x4, x5, x6, sens)

    def construct7(self, x1, x2, x3, x4, x5, x6, x7, sens):
        return self.grad(self.network)(x1, x2, x3, x4, x5, x6, x7, sens)


class InputOpNet(nn.Cell):
    """ InputOpNet definition """

    def __init__(self, op, get_first=False,
                 c1=None, c2=None, c3=None, c4=None):
        super(InputOpNet, self).__init__()
        self.op = op
        self.get_first = get_first
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4

    def construct(self, *inputs):
        pass

    def construct0_c0_fack(self, data):
        x = self.op() + data
        if self.get_first:
            x = x[0]
        return x

    def construct0_c1_fack(self, data):
        x = self.op(self.c1) + data
        if self.get_first:
            x = x[0]
        return x

    def construct0_c2_fack(self, data):
        x = self.op(self.c1, self.c2) + data
        if self.get_first:
            x = x[0]
        return x

    def construct0_c0(self):
        x = self.op()
        if self.get_first:
            x = x[0]
        return x

    def construct0_c1(self):
        x = self.op(self.c1)
        if self.get_first:
            x = x[0]
        return x

    def construct0_c2(self):
        x = self.op(self.c1, self.c2)
        if self.get_first:
            x = x[0]
        return x

    def construct1_c0(self, x1):
        x = self.op(x1)
        if self.get_first:
            x = x[0]
        return x

    def construct1_c1(self, x1):
        x = self.op(x1, self.c1)
        if self.get_first:
            x = x[0]
        return x

    def construct1_c2(self, x1):
        x = self.op(x1, self.c1, self.c2)
        if self.get_first:
            x = x[0]
        return x

    def construct1_c3(self, x1):
        x = self.op(x1, self.c1, self.c2, self.c3)
        if self.get_first:
            x = x[0]
        return x

    def construct1_c4(self, x1):
        x = self.op(x1, self.c1, self.c2, self.c3, self.c4)
        if self.get_first:
            x = x[0]
        return x

    def constructc1_1(self, x1):
        x = self.op(self.c1, x1)
        if self.get_first:
            x = x[0]
        return x

    def construct2_c0(self, x1, x2):
        x = self.op(x1, x2)
        if self.get_first:
            x = x[0]
        return x

    def construct2_c1(self, x1, x2):
        x = self.op(x1, x2, self.c1)
        if self.get_first:
            x = x[0]
        return x

    def construct2_c3(self, x1, x2):
        x = self.op(x1, x2, self.c1, self.c2, self.c3)
        if self.get_first:
            x = x[0]
        return x

    def construct3_c0(self, x1, x2, x3):
        x = self.op(x1, x2, x3)
        if self.get_first:
            x = x[0]
        return x

    def construct3_c1(self, x1, x2, x3):
        x = self.op(x1, x2, x3, self.c1)
        if self.get_first:
            x = x[0]
        return x

    def construct4_c0(self, x1, x2, x3, x4):
        x = self.op(x1, x2, x3, x4)
        if self.get_first:
            x = x[0]
        return x

    def construct4_c1(self, x1, x2, x3, x4):
        x = self.op(x1, x2, x3, x4, self.c1)
        if self.get_first:
            x = x[0]
        return x

    def construct5_c0(self, x1, x2, x3, x4, x5):
        x = self.op(x1, x2, x3, x4, x5)
        if self.get_first:
            x = x[0]
        return x

    def construct6_c0(self, x1, x2, x3, x4, x5, x6):
        x = self.op(x1, x2, x3, x4, x5, x6)
        if self.get_first:
            x = x[0]
        return x

    def construct5_c1(self, x1, x2, x3, x4, x5):
        x = self.op(x1, x2, x3, x4, x5, self.c1)
        if self.get_first:
            x = x[0]
        return x


class NetOutputAsLoss(nn.Cell):
    """ NetOutputAsLoss definition """

    def __init__(self, network, output_index):
        super(NetOutputAsLoss, self).__init__()
        self.network = network
        self.output_index = output_index

    def construct(self, *inputs):
        pass

    def construct1(self, x1):
        predict = self.network(x1)[self.output_index]
        return predict

    def construct2(self, x1, x2):
        predict = self.network(x1, x2)[self.output_index]
        return predict

    def construct3(self, x1, x2, x3):
        predict = self.network(x1, x2, x3)[self.output_index]
        return predict

    def construct4(self, x1, x2, x3, x4):
        predict = self.network(x1, x2, x3, x4)[self.output_index]
        return predict

    def construct5(self, x1, x2, x3, x4, x5):
        predict = self.network(x1, x2, x3, x4, x5)[self.output_index]
        return predict


def get_loss_fun(construct_net, num_input, output_index):
    net = NetOutputAsLoss(construct_net, output_index)
    f = getattr(net, 'construct%d' % num_input)
    setattr(net, "construct", f)
    return net


def build_construct_graph(net, *inputs, execute=True):
    net.set_train()
    _cell_graph_executor.compile(net, *inputs)
    if execute:
        _cell_graph_executor(net, inputs)


def build_backward_graph(net, output_shapes, inputs, execute=True):
    inputs = append_sens_to_inputs(output_shapes, inputs)
    net = gen_backward_net(net, len(inputs) - 1)
    net.set_train()
    _cell_graph_executor.compile(net, inputs)
    if execute:
        _cell_graph_executor(net, inputs)


def convert(shp, dtype=np.float32, scale=6):
    if isinstance(shp, list):
        if not shp:
            return Tensor((np.random.rand() * scale).astype(dtype))
        return Tensor((np.random.rand(*shp) * scale).astype(dtype))
    return shp


def gen_inputs(input_shapes, config):
    add_fack_input = config.get('add_fack_input', False)
    if not input_shapes and add_fack_input:
        return [Tensor(np.array([1.0]).astype(config.get('fack_input_type', np.float32)))]
    return [convert(shp) for shp in input_shapes]


def gen_backward_inputs(input_shapes, output_shapes, config):
    add_fack_input = config.get('add_fack_input', False)
    if not input_shapes and add_fack_input:
        inputs = [Tensor(np.array([1.0]))]
    else:
        inputs = [convert(shp) for shp in input_shapes]
    sens_shape = output_shapes[0]
    sens = convert(sens_shape)
    return inputs + [sens]


def append_sens_to_inputs(output_shapes, inputs):
    inputs = inputs
    sens = Tensor(np.random.normal(0, 1, output_shapes).astype(np.float32))
    return inputs + [sens]


def gen_net(shapes, config, get_first=False):
    """
    gen_net function
    """
    add_fack_input = config.get('add_fack_input', False)
    op = config['op']
    if 'const' not in config:
        const_input = []
    else:
        const_input = config['const']
    const_first = False
    if 'const_first' in config:
        const_first = config['const_first']

    net = InputOpNet(op, get_first, *const_input)
    if const_first:
        fn_name = 'constructc%d_%d' % (len(const_input), len(shapes))
    else:
        fn_name = 'construct%d_c%d' % (len(shapes), len(const_input))
    if add_fack_input:
        fn_name += '_fack'
    f = getattr(net, fn_name)
    setattr(net, "construct", f)
    return net


def gen_backward_net(construct_net, input_num):
    net = InputBackward(construct_net)
    f = getattr(net, 'construct%d' % input_num)
    setattr(net, "construct", f)
    return net


def batch_tuple_tensor(data, batch_size):
    ret = [Tensor(np.tile(d.asnumpy(), (batch_size, 1))) for d in data]
    return tuple(ret)


class OutPutWrap(nn.Cell):
    """
    OutPutWrap definition
    """

    def __init__(self, network, num_output, output_is_tuple):
        super(OutPutWrap, self).__init__()
        self.network = network
        self.num_output = num_output
        self.one = Tensor(np.array([1]))
        self.dtype = P.DType()
        self.cast = P.Cast()
        self.output_is_tuple = output_is_tuple

    def construct(self, *inputs):
        pass

    def construct1(self, x1):
        ret = F.make_tuple()
        predict = self.network(x1)
        if self.num_output == 1 and self.output_is_tuple == 0:
            return predict * self.cast(self.one, self.dtype(predict))
        for i in range(self.num_output):
            ret = ret + F.make_tuple(predict[i] * self.cast(self.one, self.dtype(predict[i])))
        return ret

    def construct2(self, x1, x2):
        ret = F.make_tuple()
        predict = self.network(x1, x2)
        if self.num_output == 1 and self.output_is_tuple == 0:
            return predict * self.cast(self.one, self.dtype(predict))
        for i in range(self.num_output):
            ret = ret + F.make_tuple(predict[i] * self.cast(self.one, self.dtype(predict[i])))
        return ret

    def construct3(self, x1, x2, x3):
        ret = F.make_tuple()
        predict = self.network(x1, x2, x3)
        if self.num_output == 1 and self.output_is_tuple == 0:
            return predict * self.cast(self.one, self.dtype(predict))
        for i in range(self.num_output):
            ret = ret + F.make_tuple(predict[i] * self.cast(self.one, self.dtype(predict[i])))
        return ret

    def construct4(self, x1, x2, x3, x4):
        ret = F.make_tuple()
        predict = self.network(x1, x2, x3, x4)
        if self.num_output == 1 and self.output_is_tuple == 0:
            return predict * self.cast(self.one, self.dtype(predict))
        for i in range(self.num_output):
            ret = ret + F.make_tuple(predict[i] * self.cast(self.one, self.dtype(predict[i])))
        return ret

    def construct5(self, x1, x2, x3, x4, x5):
        ret = F.make_tuple()
        predict = self.network(x1, x2, x3, x4, x5)
        if self.num_output == 1 and self.output_is_tuple == 0:
            return predict * self.cast(self.one, self.dtype(predict))
        for i in range(self.num_output):
            ret = ret + F.make_tuple(predict[i] * self.cast(self.one, self.dtype(predict[i])))
        return ret

    def construct6(self, x1, x2, x3, x4, x5, x6):
        ret = F.make_tuple()
        predict = self.network(x1, x2, x3, x4, x5, x6)
        if self.num_output == 1 and self.output_is_tuple == 0:
            return predict * self.cast(self.one, self.dtype(predict))
        for i in range(self.num_output):
            ret = ret + F.make_tuple(predict[i] * self.cast(self.one, self.dtype(predict[i])))
        return ret


def get_output_wrap(network, num_input, num_output, output_is_tuple=0):
    net = OutPutWrap(network, num_output, output_is_tuple)
    f = getattr(net, 'construct%d' % num_input)
    setattr(net, "construct", f)
    return net

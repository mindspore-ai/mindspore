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
import pytest
import numpy as np
import mindspore as ms
import mindspore.context as context
from mindspore import Tensor, Parameter
import mindspore.nn as nn
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import TrainOneStepCell, Momentum
from mindspore.ops import operations as P
from mindspore.ops.operations.comm_ops import NeighborExchange


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


_w1 = Tensor(np.ones([32, 32]), dtype=ms.float32)
_x1 = Tensor(np.ones([32, 16]), dtype=ms.float32)
_x2 = Tensor(np.ones([16, 32]), dtype=ms.float32)


def compile_net(net):
    context.set_context(mode=context.GRAPH_MODE)
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, _x1, _x2)


def test_NeighborExchange_two_inputs_success():
    """
    Feature: NeighborExchange
    Description: two inputs and two outputs, with valid arguments
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class MatMulNet(nn.Cell):
        def __init__(self, weight1):
            super(MatMulNet, self).__init__()
            self.matmul = P.MatMul()
            self.mul = P.Mul()
            self.alltoallv = NeighborExchange(send_rank_ids=[0, 1], recv_rank_ids=[1, 2],
                                              recv_shapes=([32, 32], [32, 64]),
                                              send_shapes=([32, 32], [32, 16]), recv_type=ms.float32)
            self.weight1 = Parameter(weight1, "w1")

        def construct(self, x1, x2):
            out = self.matmul(x1, x2)
            out = self.mul(out, self.weight1)
            out = self.alltoallv((out, x1))
            return out[0]

    net = MatMulNet(_w1)
    compile_net(net)


def test_NeighborExchange_single_input_success():
    """
    Feature: NeighborExchange
    Description: one inputs and two outputs, with valid arguments
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class MatMulNet2(nn.Cell):
        def __init__(self, weight1):
            super(MatMulNet2, self).__init__()
            self.matmul = P.MatMul()
            self.mul = P.Mul()
            self.alltoallv = NeighborExchange(send_rank_ids=[0], recv_rank_ids=[1, 2], recv_shapes=([32, 32], [32, 64]),
                                              send_shapes=([32, 32],), recv_type=ms.float32)
            self.weight1 = Parameter(weight1, "w1")

        def construct(self, x1, x2):
            out = self.matmul(x1, x2)
            out = self.mul(out, self.weight1)
            out = self.alltoallv((out,))
            return out[0]

    net = MatMulNet2(_w1)
    compile_net(net)


def test_NeighborExchange_empty_send_success():
    """
    Feature: NeighborExchange
    Description: empty inputs, with valid arguments
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.alltoallv = NeighborExchange(send_rank_ids=[], recv_rank_ids=[1], recv_shapes=([1],),
                                              send_shapes=(), recv_type=ms.float32)

        def construct(self, x1):
            self.alltoallv()
            return x1

    net = Net()
    _cell_graph_executor.compile(net, _x1)


def test_NeighborExchange_empty_recv_success():
    """
    Feature: NeighborExchange
    Description: empty outputs, with valid arguments
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.alltoallv = NeighborExchange(send_rank_ids=[0], recv_rank_ids=[], recv_shapes=(),
                                              send_shapes=([32, 16],), recv_type=ms.float32)

        def construct(self, x1):
            self.alltoallv((x1,))
            return x1

    net = Net()
    _cell_graph_executor.compile(net, _x1)


def test_NeighborExchange_empty_send_empty_recv_success():
    """
    Feature: NeighborExchange
    Description: empty inputs and empty outputs, with valid arguments
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.alltoallv = NeighborExchange(send_rank_ids=[], recv_rank_ids=[], recv_shapes=(),
                                              send_shapes=(), recv_type=ms.float32)

        def construct(self, x1):
            self.alltoallv()
            return x1

    net = Net()
    _cell_graph_executor.compile(net, _x1)


def test_NeighborExchange_recv_shape_num_diff_with_recv_rank_size_failed():
    """
    Feature: NeighborExchange
    Description: send_rank_ids and send_shapes are set as 1 input, but gives 2
    Expectation: throw ValueError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self, weight1):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.mul = P.Mul()
            self.alltoallv = NeighborExchange(send_rank_ids=[0], recv_rank_ids=[1, 2], recv_shapes=([32, 32],),
                                              send_shapes=([32, 32],), recv_type=ms.float32)
            self.weight1 = Parameter(weight1, "w1")

        def construct(self, x1, x2):
            out = self.matmul(x1, x2)
            out = self.mul(out, self.weight1)
            out = self.alltoallv((out,))
            return out[0]

    net = Net(_w1)
    with pytest.raises(ValueError):
        compile_net(net)


def test_NeighborExchange_send_shape_num_diff_with_send_rank_size_failed():
    """
    Feature: NeighborExchange
    Description: send_rank_ids is set as 2 inputs, but send_shapes are set as 1 input
    Expectation: throw ValueError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self, weight1):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.mul = P.Mul()
            self.alltoallv = NeighborExchange(send_rank_ids=[0, 1], recv_rank_ids=[1, 2],
                                              recv_shapes=([32, 32], [32, 32]),
                                              send_shapes=([32, 32],), recv_type=ms.float32)
            self.weight1 = Parameter(weight1, "w1")

        def construct(self, x1, x2):
            out = self.matmul(x1, x2)
            out = self.mul(out, self.weight1)
            out = self.alltoallv((out,))
            return out[0]

    net = Net(_w1)
    with pytest.raises(ValueError):
        compile_net(net)


def test_NeighborExchange_send_shape_num_diff_with_input_num_failed():
    """
    Feature: NeighborExchange
    Description: send_rank_ids and send_shapes are set as 2 inputs, but has only 1 input
    Expectation: throw Exception
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self, weight1):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.mul = P.Mul()
            self.alltoallv = NeighborExchange(send_rank_ids=[0, 1], recv_rank_ids=[1, 2],
                                              recv_shapes=([32, 32], [32, 32]),
                                              send_shapes=([32, 32], [32, 32]), recv_type=ms.float32)
            self.weight1 = Parameter(weight1, "w1")

        def construct(self, x1, x2):
            out = self.matmul(x1, x2)
            out = self.mul(out, self.weight1)
            out = self.alltoallv((out,))
            return out[0]

    net = Net(_w1)
    with pytest.raises(Exception):
        compile_net(net)


def test_NeighborExchange_send_shape_diff_with_input_shape_failed():
    """
    Feature: NeighborExchange
    Description: send_shapes is set as [16, 16], but input is [32, 32]
    Expectation: throw Exception
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self, weight1):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.mul = P.Mul()
            self.alltoallv = NeighborExchange(send_rank_ids=[0], recv_rank_ids=[1, 2], recv_shapes=([32, 32], [32, 64]),
                                              send_shapes=([16, 16],), recv_type=ms.float32)
            self.weight1 = Parameter(weight1, "w1")

        def construct(self, x1, x2):
            out = self.matmul(x1, x2)
            out = self.mul(out, self.weight1)
            out = self.alltoallv((out,))
            return out[0]

    net = Net(_w1)
    with pytest.raises(Exception):
        compile_net(net)


def test_NeighborExchange_attr_check_send_rank_ids_is_tuple_failed():
    """
    Feature: NeighborExchange
    Description: send_rank_ids should be list, but a tuple is given
    Expectation: throw TypeError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.alltoallv = NeighborExchange(send_rank_ids=(0), recv_rank_ids=[1, 2], recv_shapes=([32, 32], [32, 64]),
                                              send_shapes=([32, 16],), recv_type=ms.float32)

        def construct(self, x1):
            out = self.alltoallv((x1,))
            return out[0]

    net = Net()
    with pytest.raises(TypeError):
        _cell_graph_executor.compile(net, _x1)


def test_NeighborExchange_attr_check_send_rank_ids_is_tuple_2_failed():
    """
    Feature: NeighborExchange
    Description: send_rank_ids should be list, but a tuple is given
    Expectation: throw TypeError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.alltoallv = NeighborExchange(send_rank_ids=(0,), recv_rank_ids=[1, 2],
                                              recv_shapes=([32, 32], [32, 64]),
                                              send_shapes=([32, 16],), recv_type=ms.float32)

        def construct(self, x1):
            out = self.alltoallv((x1,))
            return out[0]

    net = Net()
    with pytest.raises(TypeError):
        _cell_graph_executor.compile(net, _x1)


def test_NeighborExchange_attr_check_send_rank_ids_is_float_failed():
    """
    Feature: NeighborExchange
    Description: send_rank_ids should be int, but a float is given
    Expectation: throw TypeError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.alltoallv = NeighborExchange(send_rank_ids=[1.0], recv_rank_ids=[1, 2],
                                              recv_shapes=([32, 32], [32, 64]),
                                              send_shapes=([32, 16],), recv_type=ms.float32)

        def construct(self, x1):
            out = self.alltoallv((x1,))
            return out[0]

    net = Net()
    with pytest.raises(TypeError):
        _cell_graph_executor.compile(net, _x1)


def test_NeighborExchange_attr_check_recv_rank_ids_is_tuple_failed():
    """
    Feature: NeighborExchange
    Description: recv_rank_ids should be list, but a tuple is given
    Expectation: throw TypeError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.alltoallv = NeighborExchange(send_rank_ids=[0], recv_rank_ids=([1, 2],),
                                              recv_shapes=([32, 32], [32, 64]),
                                              send_shapes=([32, 16],), recv_type=ms.float32)

        def construct(self, x1):
            out = self.alltoallv((x1,))
            return out[0]

    net = Net()
    with pytest.raises(TypeError):
        _cell_graph_executor.compile(net, _x1)


def test_NeighborExchange_attr_check_recv_rank_ids_is_tuple_2_failed():
    """
    Feature: NeighborExchange
    Description: recv_rank_ids should be list, but a tuple is given
    Expectation: throw TypeError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.alltoallv = NeighborExchange(send_rank_ids=[0], recv_rank_ids=(1, 2,),
                                              recv_shapes=([32, 32], [32, 64]),
                                              send_shapes=([32, 16],), recv_type=ms.float32)

        def construct(self, x1):
            out = self.alltoallv((x1,))
            return out[0]

    net = Net()
    with pytest.raises(TypeError):
        _cell_graph_executor.compile(net, _x1)


def test_NeighborExchange_attr_check_recv_rank_ids_is_float_failed():
    """
    Feature: NeighborExchange
    Description: recv_rank_ids should be int, but a float is given
    Expectation: throw TypeError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.alltoallv = NeighborExchange(send_rank_ids=[1], recv_rank_ids=[1, 2.0],
                                              recv_shapes=([32, 32], [32, 64]),
                                              send_shapes=([32, 16],), recv_type=ms.float32)

        def construct(self, x1):
            out = self.alltoallv((x1,))
            return out[0]

    net = Net()
    with pytest.raises(TypeError):
        _cell_graph_executor.compile(net, _x1)


def test_NeighborExchange_attr_check_send_shape_not_tuple_failed():
    """
    Feature: NeighborExchange
    Description: send_shapes should be tuple(list), but a list is given
    Expectation: throw TypeError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.alltoallv = NeighborExchange(send_rank_ids=[1], recv_rank_ids=[1, 2],
                                              recv_shapes=([32, 32], [32, 64]),
                                              send_shapes=([32, 16]), recv_type=ms.float32)

        def construct(self, x1):
            out = self.alltoallv((x1,))
            return out[0]

    net = Net()
    with pytest.raises(TypeError):
        _cell_graph_executor.compile(net, _x1)


def test_NeighborExchange_attr_check_send_shape_list_failed():
    """
    Feature: NeighborExchange
    Description: send_shapes should be tuple(list), but a list(list) is given
    Expectation: throw TypeError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.alltoallv = NeighborExchange(send_rank_ids=[1], recv_rank_ids=[1, 2],
                                              recv_shapes=([32, 32], [32, 64]),
                                              send_shapes=[[32, 16]], recv_type=ms.float32)

        def construct(self, x1):
            out = self.alltoallv((x1,))
            return out[0]

    net = Net()
    with pytest.raises(TypeError):
        _cell_graph_executor.compile(net, _x1)


def test_NeighborExchange_attr_check_recv_type_numpy_failed():
    """
    Feature: NeighborExchange
    Description: recv_type should be mindspore type, but a numpy type is given
    Expectation: throw TypeError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.alltoallv = NeighborExchange(send_rank_ids=[1], recv_rank_ids=[1, 2],
                                              recv_shapes=([32, 32], [32, 64]),
                                              send_shapes=([32, 16],), recv_type=np.float32)

        def construct(self, x1):
            out = self.alltoallv((x1,))
            return out[0]

    net = Net()
    with pytest.raises(TypeError):
        _cell_graph_executor.compile(net, _x1)


def test_NeighborExchange_attr_invalid_grpup_failed():
    """
    Feature: NeighborExchange
    Description: group should be str, but a tuple is given
    Expectation: throw TypeError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.alltoallv = NeighborExchange(send_rank_ids=[1], recv_rank_ids=[1, 2],
                                              recv_shapes=([32, 32], [32, 64]),
                                              send_shapes=([32, 16],), recv_type=ms.float32, group=("str",))

        def construct(self, x1):
            out = self.alltoallv((x1,))
            return out[0]

    net = Net()
    with pytest.raises(TypeError):
        _cell_graph_executor.compile(net, _x1)

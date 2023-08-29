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
from mindspore import Tensor
import mindspore.nn as nn
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import TrainOneStepCell, Momentum
from mindspore.ops.operations.comm_ops import NeighborExchangeV2

_x1 = Tensor(np.ones([1, 1, 32, 16]), dtype=ms.float32)
_x2 = Tensor(np.ones([1, 1, 33, 16]), dtype=ms.float32)


def compile_net(net, x1, x2):
    context.set_context(mode=context.GRAPH_MODE)
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, x1, x2)


def test_neighborexchangev2_single_input_success():
    """
    Feature: NeighborExchangeV2
    Description: one inputs and one outputs, with valid arguments
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.linear = nn.Dense(16, 16)
            self.neighborexchangev2 = NeighborExchangeV2(send_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         send_lens=[0, 1, 0, 0],
                                                         recv_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         recv_lens=[0, 1, 0, 0], data_format="NCHW")

        def construct(self, x1, x2):
            y = self.linear(x1)
            y = self.neighborexchangev2(y)
            y = y + x2
            return y

    net = Net()
    compile_net(net, _x1, _x2)


def test_neighborexchangev2_send_lens_equal_to_input_shape_success():
    """
    Feature: NeighborExchangeV2
    Description: send_lens is equal to input shape
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.linear = nn.Dense(16, 16)
            self.neighborexchangev2 = NeighborExchangeV2(send_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         send_lens=[0, 32, 0, 0],
                                                         recv_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         recv_lens=[0, 1, 0, 0], data_format="NCHW")

        def construct(self, x1, x2):
            y = self.linear(x1)
            y = self.neighborexchangev2(y)
            y = y + x2
            return y

    net = Net()
    compile_net(net, _x1, _x2)


def test_neighborexchangev2_empty_send_success():
    """
    Feature: NeighborExchangeV2
    Description: empty inputs, with valid arguments
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.linear = nn.Dense(16, 16)
            self.neighborexchangev2 = NeighborExchangeV2(send_rank_ids=[-1, -1, -1, -1, -1, -1, -1, -1],
                                                         send_lens=[1, 2, 3, 4],
                                                         recv_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         recv_lens=[0, 1, 0, 0],
                                                         data_format="NCHW")

        def construct(self, x1, x2):
            y = self.linear(x1)
            y = self.neighborexchangev2(y)
            y = y + x2
            return y

    net = Net()
    compile_net(net, _x1, _x2)


def test_neighborexchangev2_empty_recv_success():
    """
    Feature: NeighborExchangeV2
    Description: empty outputs, with valid arguments
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.linear = nn.Dense(16, 16)
            self.neighborexchangev2 = NeighborExchangeV2(send_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         send_lens=[0, 1, 0, 0],
                                                         recv_rank_ids=[-1, -1, -1, -1, -1, -1, -1, -1],
                                                         recv_lens=[1, 2, 3, 4],
                                                         data_format="NCHW")

        def construct(self, x1, x2):
            y = self.linear(x1)
            y = self.neighborexchangev2(y)
            y = y + x2
            return y

    net = Net()
    compile_net(net, _x1, _x1)


def test_neighborexchangev2_empty_send_empty_recv_success():
    """
    Feature: NeighborExchangeV2
    Description: empty inputs and empty outputs, with valid arguments
    Expectation: success
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.neighborexchangev2 = NeighborExchangeV2(send_rank_ids=[-1, -1, -1, -1, -1, -1, -1, -1],
                                                         send_lens=[0, 1, 0, 0],
                                                         recv_rank_ids=[-1, -1, -1, -1, -1, -1, -1, -1],
                                                         recv_lens=[1, 2, 3, 4],
                                                         data_format="NCHW")

        def construct(self, x1):
            y = self.neighborexchangev2(x1)
            return y

    net = Net()
    _cell_graph_executor.compile(net, _x1)


def test_neighborexchangev2_invalid_dataformat_failed():
    """
    Feature: NeighborExchangeV2
    Description: data_format should be NCHW, but gives NHWC
    Expectation: throw ValueError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.neighborexchangev2 = NeighborExchangeV2(send_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         send_lens=[0, 1, 0, 0],
                                                         recv_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         recv_lens=[0, 1, 0, 0],
                                                         data_format="NHWC")

        def construct(self, x):
            out = self.neighborexchangev2(x)
            return out[0]

    with pytest.raises(ValueError):
        net = Net()
        _cell_graph_executor.compile(net, _x1)


def test_neighborexchangev2_invalid_send_rank_ids_size_failed():
    """
    Feature: NeighborExchangeV2
    Description: send_rank_ids size should be 8, but gives 5
    Expectation: throw ValueError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.neighborexchangev2 = NeighborExchangeV2(send_rank_ids=[-1, -1, -1, -1, 1],
                                                         send_lens=[0, 1, 0, 0],
                                                         recv_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         recv_lens=[0, 1, 0, 0],
                                                         data_format="NCHW")

        def construct(self, x):
            out = self.neighborexchangev2(x)
            return out[0]

    with pytest.raises(ValueError):
        net = Net()
        _cell_graph_executor.compile(net, _x1)


def test_neighborexchangev2_invalid_recv_rank_ids_size_failed():
    """
    Feature: NeighborExchangeV2
    Description: recv_rank_ids size should be 8, but gives 5
    Expectation: throw ValueError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.neighborexchangev2 = NeighborExchangeV2(send_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         send_lens=[0, 1, 0, 0],
                                                         recv_rank_ids=[-1, -1, -1, -1, 1],
                                                         recv_lens=[0, 1, 0, 0],
                                                         data_format="NCHW")

        def construct(self, x):
            out = self.neighborexchangev2(x)
            return out[0]

    with pytest.raises(ValueError):
        net = Net()
        _cell_graph_executor.compile(net, _x1)


def test_neighborexchangev2_invalid_send_lens_size_failed():
    """
    Feature: NeighborExchangeV2
    Description: send_lens size should be 4, but gives 5
    Expectation: throw ValueError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.neighborexchangev2 = NeighborExchangeV2(send_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         send_lens=[0, 1, 0, 0, 2],
                                                         recv_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         recv_lens=[0, 1, 0, 0],
                                                         data_format="NCHW")

        def construct(self, x):
            out = self.neighborexchangev2(x)
            return out[0]

    with pytest.raises(ValueError):
        net = Net()
        _cell_graph_executor.compile(net, _x1)


def test_neighborexchangev2_invalid_recv_lens_size_failed():
    """
    Feature: NeighborExchangeV2
    Description: recv_lens size should be 4, but gives 5
    Expectation: throw ValueError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.neighborexchangev2 = NeighborExchangeV2(send_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         send_lens=[0, 1, 0, 0],
                                                         recv_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         recv_lens=[0, 1, 0, 0, 2],
                                                         data_format="NCHW")

        def construct(self, x):
            out = self.neighborexchangev2(x)
            return out[0]

    with pytest.raises(ValueError):
        net = Net()
        _cell_graph_executor.compile(net, _x1)


def test_neighborexchangev2_invalid_input_size_failed():
    """
    Feature: NeighborExchangeV2
    Description: input should be one tensor, but gives 2
    Expectation: throw ValueError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.neighborexchangev2 = NeighborExchangeV2(send_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         send_lens=[0, 1, 0, 0],
                                                         recv_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         recv_lens=[0, 1, 0, 0],
                                                         data_format="NCHW")

        def construct(self, x1, x2):
            out = self.neighborexchangev2(x1, x2)
            return out[0]

    with pytest.raises(ValueError):
        net = Net()
        _cell_graph_executor.compile(net, _x1, _x2)


def test_neighborexchangev2_recv_rank_ids_invalid_value_failed():
    """
    Feature: NeighborExchangeV2
    Description: recv_rank_ids should can be concat, recv_rank_ids[3] and [4] is 1, [5] is -1 given
    Expectation: throw Exception
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.neighborexchangev2 = NeighborExchangeV2(send_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         send_lens=[0, 1, 0, 0],
                                                         recv_rank_ids=[-1, -1, -1, 1, 1, -1, -1, -1],
                                                         recv_lens=[0, 1, 0, 0],
                                                         data_format="NCHW")

        def construct(self, x):
            out = self.neighborexchangev2(x)
            return out[0]

    with pytest.raises(ValueError):
        net = Net()
        _cell_graph_executor.compile(net, _x1)


def test_neighborexchangev2_attr_check_send_rank_ids_is_tuple_failed():
    """
    Feature: NeighborExchangeV2
    Description: send_rank_ids should be list, but a tuple is given
    Expectation: throw TypeError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.neighborexchangev2 = NeighborExchangeV2(send_rank_ids=(-1, -1, -1, -1, 1, -1, -1, -1),
                                                         send_lens=[0, 1, 0, 0],
                                                         recv_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         recv_lens=[0, 1, 0, 0],
                                                         data_format="NCHW")

        def construct(self, x):
            out = self.neighborexchangev2(x)
            return out[0]

    with pytest.raises(TypeError):
        net = Net()
        _cell_graph_executor.compile(net, _x1)


def test_neighborexchangev2_attr_check_send_lens_is_tuple_failed():
    """
    Feature: NeighborExchangeV2
    Description: send_lens should be list, but a tuple is given
    Expectation: throw TypeError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.neighborexchangev2 = NeighborExchangeV2(send_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         send_lens=(0, 1, 0, 0),
                                                         recv_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         recv_lens=[0, 1, 0, 0],
                                                         data_format="NCHW")

        def construct(self, x):
            out = self.neighborexchangev2(x)
            return out[0]

    with pytest.raises(TypeError):
        net = Net()
        _cell_graph_executor.compile(net, _x1)


def test_neighborexchangev2_attr_check_recv_rank_ids_is_tuple_failed():
    """
    Feature: NeighborExchangeV2
    Description: recv_rank_ids should be list, but a tuple is given
    Expectation: throw TypeError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.neighborexchangev2 = NeighborExchangeV2(send_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         send_lens=[0, 1, 0, 0],
                                                         recv_rank_ids=(-1, -1, -1, -1, 1, -1, -1, -1),
                                                         recv_lens=[0, 1, 0, 0],
                                                         data_format="NCHW")

        def construct(self, x):
            out = self.neighborexchangev2(x)
            return out[0]

    with pytest.raises(TypeError):
        net = Net()
        _cell_graph_executor.compile(net, _x1)


def test_neighborexchangev2_attr_check_recv_lens_is_tuple_failed():
    """
    Feature: NeighborExchangeV2
    Description: recv_lens should be list, but a tuple is given
    Expectation: throw TypeError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.neighborexchangev2 = NeighborExchangeV2(send_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         send_lens=[0, 1, 0, 0],
                                                         recv_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         recv_lens=(0, 1, 0, 0),
                                                         data_format="NCHW")

        def construct(self, x):
            out = self.neighborexchangev2(x)
            return out[0]

    with pytest.raises(TypeError):
        net = Net()
        _cell_graph_executor.compile(net, _x1)


def test_neighborexchangev2_attr_check_send_rank_ids_is_float_failed():
    """
    Feature: NeighborExchangeV2
    Description: send_rank_ids should be int, but float is given
    Expectation: throw TypeError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.neighborexchangev2 = NeighborExchangeV2(send_rank_ids=[-1, -1, -1, -1, 1.0, -1, -1, -1],
                                                         send_lens=[0, 1, 0, 0],
                                                         recv_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         recv_lens=[0, 1, 0, 0],
                                                         data_format="NCHW")

        def construct(self, x):
            out = self.neighborexchangev2(x)
            return out[0]

    with pytest.raises(TypeError):
        net = Net()
        _cell_graph_executor.compile(net, _x1)


def test_neighborexchangev2_attr_check_send_lens_is_float_failed():
    """
    Feature: NeighborExchangeV2
    Description: send_lens should be int, but float is given
    Expectation: throw TypeError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.neighborexchangev2 = NeighborExchangeV2(send_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         send_lens=[0, 1.0, 0, 0],
                                                         recv_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         recv_lens=[0, 1, 0, 0],
                                                         data_format="NCHW")

        def construct(self, x):
            out = self.neighborexchangev2(x)
            return out[0]

    with pytest.raises(TypeError):
        net = Net()
        _cell_graph_executor.compile(net, _x1)


def test_neighborexchangev2_attr_check_recv_rank_ids_is_float_failed():
    """
    Feature: NeighborExchangeV2
    Description: send_rank_ids should be int, but float is given
    Expectation: throw TypeError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.neighborexchangev2 = NeighborExchangeV2(send_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         send_lens=[0, 1, 0, 0],
                                                         recv_rank_ids=[-1, -1, -1, -1, 1.0, -1, -1, -1],
                                                         recv_lens=[0, 1, 0, 0],
                                                         data_format="NCHW")

        def construct(self, x):
            out = self.neighborexchangev2(x)
            return out[0]

    with pytest.raises(TypeError):
        net = Net()
        _cell_graph_executor.compile(net, _x1)


def test_neighborexchangev2_attr_check_recv_lens_is_float_failed():
    """
    Feature: NeighborExchangeV2
    Description: ids in send_rank_ids should be int, but float is given
    Expectation: throw TypeError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.neighborexchangev2 = NeighborExchangeV2(send_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         send_lens=[0, 1, 0, 0],
                                                         recv_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         recv_lens=[0, 1.0, 0, 0],
                                                         data_format="NCHW")

        def construct(self, x):
            out = self.neighborexchangev2(x)
            return out[0]

    with pytest.raises(TypeError):
        net = Net()
        _cell_graph_executor.compile(net, _x1)


def test_neighborexchangev2_group_is_tuple_failed():
    """
    Feature: NeighborExchangeV2
    Description: group should be a string, but tuple given
    Expectation: throw TypeError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.neighborexchangev2 = NeighborExchangeV2(send_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         send_lens=[0, 1, 0, 0],
                                                         recv_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         recv_lens=[0, 1, 0, 0],
                                                         data_format="NCHW", group=("str",))

        def construct(self, x):
            out = self.neighborexchangev2(x)
            return out[0]

    with pytest.raises(TypeError):
        net = Net()
        _cell_graph_executor.compile(net, _x1)


def test_neighborexchangev2_send_lens_larger_than_input_shape_failed():
    """
    Feature: NeighborExchangeV2
    Description: send_lens should be <= input_shape, but a larger one given
    Expectation: throw TypeError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.neighborexchangev2 = NeighborExchangeV2(send_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         send_lens=[0, 35, 0, 0],
                                                         recv_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         recv_lens=[0, 1, 0, 0],
                                                         data_format="NCHW")

        def construct(self, x):
            out = self.neighborexchangev2(x)
            return out[0]

    with pytest.raises(ValueError):
        net = Net()
        _cell_graph_executor.compile(net, _x1)


def test_neighborexchangev2_send_rank_ids_value_invalid_failed():
    """
    Feature: NeighborExchangeV2
    Description: send_rank_ids should be >=0 or -1, but -3 is given
    Expectation: throw TypeError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.neighborexchangev2 = NeighborExchangeV2(send_rank_ids=[-1, -1, -1, -3, 1, -1, -1, -1],
                                                         send_lens=[0, 1, 0, 0],
                                                         recv_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         recv_lens=[0, 1, 0, 0],
                                                         data_format="NCHW")

        def construct(self, x):
            out = self.neighborexchangev2(x)
            return out[0]

    with pytest.raises(ValueError):
        net = Net()
        _cell_graph_executor.compile(net, _x1)


def test_neighborexchangev2_recv_rank_ids_value_invalid_failed():
    """
    Feature: NeighborExchangeV2
    Description: recv_rank_ids should be >=0 or -1, but -3 is given
    Expectation: throw TypeError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.neighborexchangev2 = NeighborExchangeV2(send_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         send_lens=[0, 1, 0, 0],
                                                         recv_rank_ids=[-1, -1, -1, -3, 1, -1, -1, -1],
                                                         recv_lens=[0, 1, 0, 0],
                                                         data_format="NCHW")

        def construct(self, x):
            out = self.neighborexchangev2(x)
            return out[0]

    with pytest.raises(ValueError):
        net = Net()
        _cell_graph_executor.compile(net, _x1)


def test_neighborexchangev2_send_lens_value_invalid_failed():
    """
    Feature: NeighborExchangeV2
    Description: send_lens should be >=0, but -3 is given
    Expectation: throw TypeError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.neighborexchangev2 = NeighborExchangeV2(send_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         send_lens=[0, -3, 0, 0],
                                                         recv_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         recv_lens=[0, 1, 0, 0],
                                                         data_format="NCHW")

        def construct(self, x):
            out = self.neighborexchangev2(x)
            return out[0]

    with pytest.raises(ValueError):
        net = Net()
        _cell_graph_executor.compile(net, _x1)


def test_neighborexchangev2_recv_lens_value_invalid_failed():
    """
    Feature: NeighborExchangeV2
    Description: recv_lens should be >=0, but -3 is given
    Expectation: throw TypeError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.neighborexchangev2 = NeighborExchangeV2(send_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         send_lens=[0, 1, 0, 0],
                                                         recv_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         recv_lens=[0, -3, 0, 0],
                                                         data_format="NCHW")

        def construct(self, x):
            out = self.neighborexchangev2(x)
            return out[0]

    with pytest.raises(ValueError):
        net = Net()
        _cell_graph_executor.compile(net, _x1)


def test_neighborexchangev2_send_rank_ids_repeat_failed():
    """
    Feature: NeighborExchangeV2
    Description: send_rank_ids cannot be repeated, but two 1 is given
    Expectation: throw TypeError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.neighborexchangev2 = NeighborExchangeV2(send_rank_ids=[1, -1, -1, -1, 1, -1, -1, -1],
                                                         send_lens=[0, 1, 0, 0],
                                                         recv_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         recv_lens=[0, 1, 0, 0],
                                                         data_format="NCHW")

        def construct(self, x):
            out = self.neighborexchangev2(x)
            return out[0]

    with pytest.raises(ValueError):
        net = Net()
        _cell_graph_executor.compile(net, _x1)


def test_neighborexchangev2_recv_rank_ids_repeat_failed():
    """
    Feature: NeighborExchangeV2
    Description: recv_rank_ids cannot be repeated, but two 1 is given
    Expectation: throw TypeError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.neighborexchangev2 = NeighborExchangeV2(send_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         send_lens=[0, 1, 0, 0],
                                                         recv_rank_ids=[1, -1, -1, -1, 1, -1, -1, -1],
                                                         recv_lens=[0, 1, 0, 0],
                                                         data_format="NCHW")

        def construct(self, x):
            out = self.neighborexchangev2(x)
            return out[0]

    with pytest.raises(ValueError):
        net = Net()
        _cell_graph_executor.compile(net, _x1)


def test_neighborexchangev2_recv_rank_ids_num_bigger_than_device_num_failed():
    """
    Feature: NeighborExchangeV2
    Description: recv_rank_ids cannot bigger than device num, but got 8
    Expectation: throw TypeError
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, dataset_strategy="data_parallel")

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.neighborexchangev2 = NeighborExchangeV2(send_rank_ids=[-1, -1, -1, -1, 1, -1, -1, -1],
                                                         send_lens=[0, 1, 0, 0],
                                                         recv_rank_ids=[8, -1, -1, -1, 1, -1, -1, -1],
                                                         recv_lens=[0, 1, 0, 0],
                                                         data_format="NCHW")

        def construct(self, x):
            out = self.neighborexchangev2(x)
            return out[0]

    with pytest.raises(ValueError):
        net = Net()
        _cell_graph_executor.compile(net, _x1)

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
import numpy as np
import pytest

import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from mindspore.ops import operations as P
from parallel.utils.utils import ParallelValidator


class Net(Cell):
    def __init__(self, weight, w2, begin, end, strides, strategy1=None, strategy2=None, is_parameter=True,
                 begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        self.strided_slice = P.StridedSlice(begin_mask=begin_mask,
                                            end_mask=end_mask,
                                            ellipsis_mask=ellipsis_mask, new_axis_mask=new_axis_mask,
                                            shrink_axis_mask=shrink_axis_mask).shard(strategy2)
        if is_parameter:
            self.weight = Parameter(weight, "w1")
        else:
            self.weight = weight
        self.mul2 = P.Mul()
        self.weight2 = Parameter(w2, "w2")
        self.begin = begin
        self.end = end
        self.strides = strides

    def construct(self, x, b):
        out = self.strided_slice(self.weight, self.begin, self.end, self.strides)
        out = self.mul(x, out)
        out = self.mul2(out, self.weight2)
        return out


class Net2(Cell):
    def __init__(self, weight2, begin, end, strides, strategy1=None, strategy2=None,
                 begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        self.strided_slice = P.StridedSlice(begin_mask=begin_mask,
                                            end_mask=end_mask,
                                            ellipsis_mask=ellipsis_mask, new_axis_mask=new_axis_mask,
                                            shrink_axis_mask=shrink_axis_mask).shard(strategy2)
        self.weight2 = Parameter(weight2, "w2")
        self.begin = begin
        self.end = end
        self.strides = strides

    def construct(self, x, b):
        out = self.mul(x, self.weight2)
        out = self.strided_slice(out, self.begin, self.end, self.strides)
        return out


_x1 = Tensor(np.ones([128, 64, 1]), dtype=ms.float32)
_x2 = Tensor(np.ones([1, 64, 32, 32]), dtype=ms.float32)
_x3 = Tensor(np.ones([64, 32]), dtype=ms.float32)
_w1 = Tensor(np.ones([256, 64, 32]), dtype=ms.float32)
_w2 = Tensor(np.ones([128, 64, 1]), dtype=ms.float32)
_w3 = Tensor(np.ones([1, 64, 32, 32]), dtype=ms.float32)
_b1 = Tensor(np.ones([128, 64, 32]), dtype=ms.float32)
_b2 = Tensor(np.ones([1, 64, 32, 32]), dtype=ms.float32)


def compile_net(net, _x1, _b1):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_auto_parallel()
    train_net.set_train()
    _cell_graph_executor.compile(train_net, _x1, _b1)
    context.reset_auto_parallel_context()


def compile_net_utils(net: Cell, *inputs):
    net.set_auto_parallel()
    net.set_train()
    phase, _ = _cell_graph_executor.compile(net, *inputs, auto_parallel_mode=True)
    context.reset_auto_parallel_context()
    return phase


def test_stridedslice_no_fully_fetch_split_error():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 2), (2, 2, 2))
    strategy2 = ((2, 2, 2),)
    net = Net(_w1, _w2, (0, 0, 0), (128, 64, 32), (1, 1, 1), strategy1, strategy2, is_parameter=True)
    with pytest.raises(RuntimeError):
        compile_net(net, _x1, _b1)


def test_stridedslice_strides_no_1_split_error():
    """
    Feature: distribute operator stridedslice in auto parallel mode.
    Description: test stridedslice with strides no 1 split in semi auto parallel.
    Expectation: compile error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 2), (2, 2, 2))
    strategy2 = ((1, 2, 2),)
    net = Net(_w1, _w2, (0, 0, 0), (128, 64, 32), (1, 1, 2), strategy1, strategy2, is_parameter=True)
    with pytest.raises(RuntimeError):
        compile_net(net, _x1, _b1)


def test_stridedslice_begin_size_smaller():
    """
    Feature: distribute operator stridedslice in auto parallel mode.
    Description: test stridedslice with begin size is smaller in semi auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 4, 1), (1, 4, 2))
    strategy2 = ((1, 4, 2),)
    net = Net(_w1, _w2, (0, 0), (128, 64), (1, 1), strategy1, strategy2, is_parameter=True)
    compile_net(net, _x1, _b1)


def test_stridedslice_parameter():
    """
    Feature: distribute operator stridedslice in auto parallel mode.
    Description: test stridedslice of parameter in semi auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 4, 1), (1, 4, 2))
    strategy2 = ((1, 4, 2),)
    net = Net(_w1, _w2, (0, 0, 0), (128, 64, 32), (1, 1, 1), strategy1, strategy2, is_parameter=True)
    compile_net(net, _x1, _b1)


def test_stridedslice_begin_mask_no_0_split_parameter():
    """
    Feature: distribute operator stridedslice in auto parallel mode.
    Description: test stridedslice with begin mask no 0 split in semi auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 4, 1), (1, 4, 2))
    strategy2 = ((1, 4, 2),)
    net = Net(_w1, _w2, (0, 0, 0), (128, 64, 32), (1, 1, 1), strategy1, strategy2, is_parameter=True, begin_mask=1)
    compile_net(net, _x1, _b1)


def test_stridedslice_end_mask_no_0_parameter():
    """
    Feature: distribute operator stridedslice in auto parallel mode.
    Description: test stridedslice with end mask no 0 in semi auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 4, 1), (1, 4, 2))
    strategy2 = ((1, 4, 2),)
    net = Net(_w1, _w2, (127, 0, 0), (128, 63, 32), (1, 1, 1), strategy1, strategy2, is_parameter=True,
              begin_mask=1, end_mask=2)
    compile_net(net, _x1, _b1)


def test_stridedslice_ellipsis_mask_no_0_parameter():
    """
    Feature: distribute operator stridedslice in auto parallel mode.
    Description: test stridedslice with ellipsis mask no 0 in semi auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 4, 1), (1, 4, 2))
    strategy2 = ((1, 4, 2),)
    net = Net(_w1, _w2, (127, 0, 0), (128, 64, 32), (1, 1, 1), strategy1, strategy2, is_parameter=True,
              begin_mask=1, end_mask=2, ellipsis_mask=4)
    compile_net(net, _x1, _b1)


def test_stridedslice_new_axis_mask_no_0_parameter():
    """
    Feature: distribute operator stridedslice in auto parallel mode.
    Description: test stridedslice with new axis mask no 0 in semi auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 4, 2, 1), (1, 4, 2, 1))
    strategy2 = ((1, 1, 4),)
    net = Net(_w1, _w3, (0, 0, 0), (128, 64, 32), (1, 1, 1), strategy1, strategy2, is_parameter=True,
              new_axis_mask=1)
    compile_net(net, _x2, _b2)


def test_stridedslice_shrink_axis_mask_no_0_parameter():
    """
    Feature: distribute operator stridedslice in auto parallel mode.
    Description: test stridedslice with shrink axis mask no 0 in semi auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 2), (1, 2))
    strategy2 = ((1, 4, 1),)
    net = Net(_w1, _w2, (0, 0, 0), (128, 64, 32), (1, 1, 1), strategy1, strategy2, is_parameter=True,
              shrink_axis_mask=1)
    compile_net(net, _x3, _b1)


def test_stridedslice_tensor():
    """
    Feature: distribute operator stridedslice in auto parallel mode.
    Description: test stridedslice of tensor in semi auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 4, 1), (1, 4, 2))
    strategy2 = ((1, 4, 2),)
    net = Net(_w1, _w2, (0, 0, 0), (128, 64, 32), (1, 1, 1), strategy1, strategy2, is_parameter=False)
    compile_net(net, _x1, _b1)


def test_stridedslice_begin_mask_no_0_tensor():
    """
    Feature: distribute operator stridedslice in auto parallel mode.
    Description: test stridedslice with begin mask no 0 in semi auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 4, 1), (1, 4, 2))
    strategy2 = ((1, 4, 2),)
    net = Net(_w1, _w2, (127, 0, 0), (128, 64, 32), (1, 1, 1), strategy1, strategy2, is_parameter=False, begin_mask=1)
    compile_net(net, _x1, _b1)


def test_stridedslice_end_mask_no_0_tensor():
    """
    Feature: distribute operator stridedslice in auto parallel mode.
    Description: test stridedslice with end mask no 0 in semi auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 4, 1), (1, 4, 2))
    strategy2 = ((1, 4, 2),)
    net = Net(_w1, _w2, (0, 0, 0), (128, 63, 32), (1, 1, 1), strategy1, strategy2, is_parameter=False, end_mask=2)
    compile_net(net, _x1, _b1)


def test_stridedslice_ellipsis_mask_no_0_tensor():
    """
    Feature: distribute operator stridedslice in auto parallel mode.
    Description: test stridedslice with ellipsis mask no 0 in semi auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 4, 1), (1, 4, 2))
    strategy2 = ((1, 4, 2),)
    net = Net(_w1, _w2, (127, 0, 0), (128, 64, 32), (1, 1, 1), strategy1, strategy2, is_parameter=False,
              begin_mask=1, end_mask=2, ellipsis_mask=4)
    compile_net(net, _x1, _b1)


def test_stridedslice_new_axis_mask_no_0_tensor():
    """
    Feature: distribute operator stridedslice in auto parallel mode.
    Description: test stridedslice with new axis mask no 0 in semi auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 4, 2, 1), (1, 4, 2, 1))
    strategy2 = ((1, 1, 4),)
    net = Net(_w1, _w3, (0, 0, 0), (128, 64, 32), (1, 1, 1), strategy1, strategy2, is_parameter=False,
              new_axis_mask=1)
    compile_net(net, _x2, _b2)


def test_stridedslice_shrink_axis_mask_no_0_tensor():
    """
    Feature: distribute operator stridedslice in auto parallel mode.
    Description: test stridedslice with shrink axis mask no 0 in semi auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 2), (1, 2))
    strategy2 = ((1, 4, 1),)
    net = Net(_w1, _w2, (0, 0, 0), (128, 64, 32), (1, 1, 1), strategy1, strategy2, is_parameter=False,
              shrink_axis_mask=1)
    compile_net(net, _x3, _b1)


def test_stridedslice_parameter_no_full_split():
    """
    Feature: distribute operator stridedslice in auto parallel mode.
    Description: test stridedslice with no full split in semi auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 4, 1), (1, 4, 2))
    strategy2 = ((1, 2, 2),)
    net = Net(_w1, _w2, (0, 0, 0), (128, 64, 32), (1, 1, 1), strategy1, strategy2, is_parameter=True)
    compile_net(net, _x1, _b1)


def test_stridedslice_output():
    """
    Feature: distribute operator stridedslice in auto parallel mode.
    Description: test stridedslice of output in semi auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 8, 1), (1, 8, 1))
    strategy2 = ((1, 8, 1),)
    net = Net2(_w2, (0, 0, 0), (64, 64, 1), (1, 1, 1), strategy1, strategy2)
    compile_net(net, _x1, _b1)


def test_stridedslice_begin_mask_no_0_output():
    """
    Feature: distribute operator stridedslice in auto parallel mode.
    Description: test stridedslice with begin mask no 0 in semi auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 8, 1), (1, 8, 1))
    strategy2 = ((1, 8, 1),)
    net = Net2(_w2, (61, 0, 0), (64, 64, 1), (1, 1, 1), strategy1, strategy2, begin_mask=1)
    compile_net(net, _x1, _b1)


def test_stridedslice_end_mask_no_0_output():
    """
    Feature: distribute operator stridedslice in auto parallel mode.
    Description: test stridedslice with end mask no 0 in semi auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 8, 1), (1, 8, 1))
    strategy2 = ((1, 8, 1),)
    net = Net2(_w2, (0, 0, 0), (64, 63, 1), (1, 1, 1), strategy1, strategy2, end_mask=2)
    compile_net(net, _x1, _b1)


def test_stridedslice_ellipsis_mask_no_0_output():
    """
    Feature: distribute operator stridedslice in auto parallel mode.
    Description: test stridedslice with ellipsis mask no 0 in semi auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 8, 1), (1, 8, 1))
    strategy2 = ((1, 8, 1),)
    net = Net2(_w2, (63, 0, 0), (64, 63, 1), (1, 1, 1), strategy1, strategy2,
               begin_mask=1, end_mask=2, ellipsis_mask=4)
    compile_net(net, _x1, _b1)


def test_stridedslice_new_axis_mask_no_0_output():
    """
    Feature: distribute operator stridedslice in auto parallel mode.
    Description: test stridedslice with new axis mask no 0 in semi auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 8, 1), (1, 8, 1))
    strategy2 = ((8, 1, 1),)
    net = Net2(_w2, (0, 0, 0), (64, 64, 1), (1, 1, 1), strategy1, strategy2, new_axis_mask=1)
    compile_net(net, _x1, _b1)


def test_stridedslice_shrink_axis_mask_no_0_output():
    """
    Feature: distribute operator stridedslice in auto parallel mode.
    Description: test stridedslice with shrink axis mask no 0 in semi auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 8, 1), (1, 8, 1))
    strategy2 = ((1, 8, 1),)
    net = Net2(_w2, (0, 0, 0), (64, 64, 1), (1, 1, 1), strategy1, strategy2, shrink_axis_mask=1)
    compile_net(net, _x1, _b1)


def test_stridedslice_output_no_full_split():
    """
    Feature: distribute operator stridedslice in auto parallel mode.
    Description: test stridedslice with no full split in semi auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 8, 1), (1, 8, 1))
    strategy2 = ((1, 4, 1),)
    net = Net2(_w2, (0, 0, 0), (64, 64, 1), (1, 1, 1), strategy1, strategy2)
    compile_net(net, _x1, _b1)


def test_stridedslice_no_strategy():
    """
    Feature: distribute operator stridedslice in auto parallel mode.
    Description: test stridedslice with no strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 8, 1), (1, 8, 1))
    strategy2 = None
    net = Net2(_w2, (0, 0, 0), (128, 64, 1), (1, 1, 1), strategy1, strategy2)
    compile_net(net, _x1, _b1)


def test_stridedslice_begin_mask_no_0_no_strategy():
    """
    Feature: distribute operator stridedslice in auto parallel mode.
    Description: test stridedslice with begin mask no 0 in auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 8, 1), (1, 8, 1))
    strategy2 = None
    net = Net2(_w2, (127, 0, 0), (128, 64, 1), (1, 1, 1), strategy1, strategy2, begin_mask=1)
    compile_net(net, _x1, _b1)


def test_stridedslice_auto_parallel():
    """
    Feature: distribute operator stridedslice in auto parallel mode.
    Description: test stridedslice in auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = Net2(_w2, (0, 0, 0), (32, 64, 1), (1, 1, 1))
    compile_net(net, _x1, _b1)



def test_stridedslice_begin_mask_no_0_auto_parallel():
    """
    Feature: distribute operator stridedslice in auto parallel mode.
    Description: test stridedslice with begin mask no 0 in auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = Net2(_w2, (29, 0, 0), (32, 64, 1), (1, 1, 1), begin_mask=1)
    compile_net(net, _x1, _b1)


def test_stridedslice_layout():
    """
    Features: StridedSlice
    Description: validate layout and structure
    Expectation: No raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 4, 1), (1, 4, 2))
    strategy2 = ((1, 4, 2),)
    net = Net(_w1, _w2, (127, 0, 0), (128, 64, 32), (1, 1, 1), strategy1, strategy2, is_parameter=True,
              begin_mask=1, end_mask=2, ellipsis_mask=4)
    phase = compile_net_utils(net, _x1, _b1)
    validator = ParallelValidator(net, phase)

    # check layout
    features_expect_layout = ([4, 2], [-1, 1, 0], [256, 16, 16], 0, True, '')
    assert validator.check_parameter_layout('w1', features_expect_layout)

    # check attrs
    roi_expect_attrs = {'begin_mask': 1, 'end_mask': 2, 'ellipsis_mask': 4}
    assert validator.check_node_attrs('StridedSlice-1', roi_expect_attrs)

    # check inputs
    roi_expect_inputs = ['Load-0', 'out((127, 0, 0))', 'out((128, 64, 32))', 'out((1, 1, 1))']
    assert validator.check_node_inputs('StridedSlice-1', roi_expect_inputs)

    # check sub_graph
    sub_graph = {
        'StridedSlice-1': ['Load-0', 'out((127, 0, 0))', 'out((128, 64, 32))', 'out((1, 1, 1))'],
        'Mul-0': ['Reshape-1', 'StridedSlice-1'],
        'AllGather-2': ['Reshape-2'],
        'Split-1': ['AllGather-2'],
        'TupleGetItem-3': ['Split-1', 0],
        'TupleGetItem-4': ['Split-1', 1],
        'TupleGetItem-5': ['Split-1', 2],
        'TupleGetItem-6': ['Split-1', 3],
        'MakeTuple-2': ['TupleGetItem-3', 'TupleGetItem-4', 'TupleGetItem-5', 'TupleGetItem-6'],
        'Concat-1': ['MakeTuple-2']
    }
    assert validator.check_graph_structure(sub_graph)

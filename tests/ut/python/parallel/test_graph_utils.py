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
"""Test graph_utils.cc model"""
import numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Symbol
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.wrap.cell_wrapper import _VirtualDatasetCell
from mindspore import context, Tensor, Parameter
from mindspore.context import ParallelMode
from parallel.utils.utils import compile_net, ParallelValidator

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


def setup_function():
    context.reset_auto_parallel_context()


def test_no_need_to_accomplish():
    """
    Feature: Accomplish partial shape in dynamic shape scene.
    Description: Test no need to accomplish partial shape.
    Expectation: Compile success.
    """

    class NoNeedToAccomplish(nn.Cell):
        def __init__(self, from_shard):
            super(NoNeedToAccomplish, self).__init__()
            self.relu = P.ReLU().shard((from_shard,))

        def construct(self, x):
            out = x
            shape = F.shape(out)
            out = F.reshape(out, (shape[0] * shape[1], shape[2] * shape[3]))
            out = self.relu(out)
            return out

    context.set_context(save_graphs=True, save_graphs_path="./test_no_need_to_accomplish")
    dataset_shard = (1, 1, 1, 1)
    from_shard = (2, 2)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=4,
                                      dataset_strategy=(dataset_shard,))
    model = NoNeedToAccomplish(from_shard=from_shard).to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    input_ids = Tensor(shape=[4, None, 2, 2], dtype=mstype.float16)
    model.set_inputs(input_ids)
    phase = compile_net(model, input_ids)
    validator = ParallelValidator(model, phase)
    assert validator.check_node_inputs_has('TupleGetItem-0', ['MakeTuple-0', 0])
    assert validator.check_node_inputs_has('Split-0', ['TupleGetItem-0', 0, 2])
    assert validator.check_node_inputs_has('TupleGetItem-1', ['Split-0', 0])
    assert validator.check_node_inputs_has('Split-1', ['TupleGetItem-1', 2, 2])
    assert validator.check_node_inputs_has('TupleGetItem-2', ['Split-1', 0])
    assert validator.check_node_inputs_has('Reshape-0', ['TupleGetItem-2', 'ValueNode_48((-1, 2))'])
    assert validator.check_node_inputs_has('ReLU-0', ['Reshape-0'])


def test_no_need_to_accomplish_static():
    """
    Feature: Accomplish partial shape in dynamic shape scene.
    Description: Test no need to accomplish partial shape.
    Expectation: Compile success.
    """

    class NoNeedToAccomplish(nn.Cell):
        def __init__(self, from_shard):
            super(NoNeedToAccomplish, self).__init__()
            self.relu = P.ReLU().shard((from_shard,))

        def construct(self, x):
            out = x
            shape = F.shape(out)
            out = F.reshape(out, (shape[0] * shape[1], shape[2] * shape[3]))
            out = self.relu(out)
            return out

    context.set_context(save_graphs=True, save_graphs_path="./test_no_need_to_accomplish_static")
    dataset_shard = (1, 1, 1, 1)
    from_shard = (2, 2)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=4,
                                      dataset_strategy=(dataset_shard,))
    model = NoNeedToAccomplish(from_shard=from_shard).to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    input_ids = Tensor(np.random.rand(4, 3, 2, 2), dtype=mstype.float16)
    model.set_inputs(input_ids)
    _ = compile_net(model, input_ids)


def test_shape_used_by_one_rank0():
    """
    Feature: Accomplish partial shape in dynamic shape scene.
    Description: Test shape used by one successor in dynamic shape.
    Expectation: Compile success.
    """

    class ShapeUsedByOneOp(nn.Cell):
        def __init__(self, previous_shard, successor_shard):
            super(ShapeUsedByOneOp, self).__init__()
            bias_shard = (1, 1, 1, 1)
            self.bias = Tensor(np.array([1]).reshape(1, 1, 1, 1))
            self.add = P.Add().shard((previous_shard, bias_shard))
            self.relu = P.ReLU().shard((successor_shard,))

        def construct(self, x):
            out = x
            out = self.add(out, self.bias)  # 1,2,1,1
            shape = F.shape(out)
            out = F.reshape(out, (shape[0] * shape[1], shape[2], shape[3]))
            out = self.relu(out)  # 2,2,2
            return out

    context.set_context(save_graphs=True, save_graphs_path="./test_shape_used_by_one_rank0")
    dataset_shard = (1, 1, 1, 1)
    successor_shard = (2, 2, 2)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8,
                                      dataset_strategy=(dataset_shard,))
    model = ShapeUsedByOneOp((1, 2, 1, 1), successor_shard).to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    d1 = Symbol(divisor=2)
    input_ids = Tensor(shape=[4, d1, 2, 2], dtype=mstype.float16)
    model.set_inputs(input_ids)
    phase = compile_net(model, input_ids)
    validator = ParallelValidator(model, phase)
    assert validator.check_node_inputs_has('MakeTuple-0', ['inputs0'])
    assert validator.check_node_inputs_has('TupleGetItem-0', ['MakeTuple-0', 0])
    assert validator.check_node_inputs_has('Split-0', ['TupleGetItem-0', 1, 2])
    assert validator.check_node_inputs_has('TupleGetItem-1', ['Split-0', 0])
    assert validator.check_node_inputs_has('Add-0', ['TupleGetItem-1', '_GetTensorSlice-0'])
    assert validator.check_node_inputs_has('Split-1', ['Add-0', 3, 2])
    assert validator.check_node_inputs_has('TupleGetItem-2', ['Split-1', 0])
    assert validator.check_node_inputs_has('Split-2', ['TupleGetItem-2', 2, 2])
    assert validator.check_node_inputs_has('TupleGetItem-3', ['Split-2', 0])
    assert validator.check_node_inputs_has('AllGather-0', ['TupleGetItem-3'])
    assert validator.check_node_inputs_has('Split-3', ['AllGather-0', 0, 2])
    assert validator.check_node_inputs_has('TupleGetItem-4', ['Split-3', 0])
    assert validator.check_node_inputs_has('TupleGetItem-5', ['Split-3', 1])
    assert validator.check_node_inputs_has('MakeTuple-1', ['TupleGetItem-4', 'TupleGetItem-5'])
    assert validator.check_node_inputs_has('Concat-0', ['MakeTuple-1', 1])
    assert validator.check_node_inputs_has('Split-4', ['Concat-0', 0, 2])
    assert validator.check_node_inputs_has('TupleGetItem-6', ['Split-4', 0])
    assert validator.check_node_inputs_fuzzy_match('Reshape-0', ['TupleGetItem-6', '(-1, 1, 1)'])
    assert validator.check_node_inputs_has('ReLU-0', ['Reshape-0'])


def test_shape_used_by_one_rank6():
    """
    Feature: Accomplish partial shape in dynamic shape scene.
    Description: Test shape used by one successor in dynamic shape on rank6.
    Expectation: Compile success.
    """

    class ShapeUsedByOneOp(nn.Cell):
        def __init__(self, previous_shard, successor_shard):
            super(ShapeUsedByOneOp, self).__init__()
            bias_shard = (1, 1, 1, 1)
            self.bias = Tensor(np.array([1]).reshape(1, 1, 1, 1))
            self.add = P.Add().shard((previous_shard, bias_shard))
            self.relu = P.ReLU().shard((successor_shard,))

        def construct(self, x):
            out = x
            out = self.add(out, self.bias)  # 1,2,1,1
            shape = F.shape(out)
            out = F.reshape(out, (shape[0] * shape[1], shape[2], shape[3]))
            out = self.relu(out)  # 2,2,2
            return out

    context.set_context(save_graphs=True, save_graphs_path="./test_shape_used_by_one_rank6")
    dataset_shard = (1, 1, 1, 1)
    successor_shard = (2, 2, 2)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=6, device_num=8,
                                      dataset_strategy=(dataset_shard,))
    model = ShapeUsedByOneOp((1, 2, 1, 1), successor_shard).to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    d1 = Symbol(divisor=2)
    input_ids = Tensor(shape=[4, d1, 2, 2], dtype=mstype.float16)
    model.set_inputs(input_ids)
    phase = compile_net(model, input_ids)
    validator = ParallelValidator(model, phase)
    assert validator.check_node_inputs_has('TupleGetItem-0', ['MakeTuple-0', 0])
    assert validator.check_node_inputs_has('Split-0', ['TupleGetItem-0', 1, 2])
    assert validator.check_node_inputs_has('TupleGetItem-1', ['Split-0', 1])
    assert validator.check_node_inputs_has('Add-0', ['TupleGetItem-1', '_GetTensorSlice-0'])
    assert validator.check_node_inputs_has('Split-1', ['Add-0', 3, 2])
    assert validator.check_node_inputs_has('TupleGetItem-2', ['Split-1', 0])
    assert validator.check_node_inputs_has('Split-2', ['TupleGetItem-2', 2, 2])
    assert validator.check_node_inputs_has('TupleGetItem-3', ['Split-2', 1])
    assert validator.check_node_inputs_has('AllGather-0', ['TupleGetItem-3'])
    assert validator.check_node_inputs_has('Split-3', ['AllGather-0', 0, 2])
    assert validator.check_node_inputs_has('TupleGetItem-4', ['Split-3', 0])
    assert validator.check_node_inputs_has('TupleGetItem-5', ['Split-3', 1])
    assert validator.check_node_inputs_has('MakeTuple-1', ['TupleGetItem-4', 'TupleGetItem-5'])
    assert validator.check_node_inputs_has('Concat-0', ['MakeTuple-1', 1])
    assert validator.check_node_inputs_has('Split-4', ['Concat-0', 0, 2])
    assert validator.check_node_inputs_has('TupleGetItem-6', ['Split-4', 1])
    assert validator.check_node_inputs_fuzzy_match('Reshape-0', ['TupleGetItem-6', '(-1, 1, 1)'])
    assert validator.check_node_inputs_has('ReLU-0', ['Reshape-0'])
    assert validator.check_node_inputs_has('Return-0', ['ReLU-0'])


def test_shape_used_by_one_static():
    """
    Feature: Accomplish partial shape in dynamic shape scene.
    Description: Test shape used by one successor in dynamic shape.
    Expectation: Compile success.
    """

    class ShapeUsedByOneOp(nn.Cell):
        def __init__(self, previous_shard, successor_shard):
            super(ShapeUsedByOneOp, self).__init__()
            bias_shard = (1, 1, 1, 1)
            self.bias = Tensor(np.array([1]).reshape(1, 1, 1, 1))
            self.add = P.Add().shard((previous_shard, bias_shard))
            self.relu = P.ReLU().shard((successor_shard,))

        def construct(self, x):
            out = x
            out = self.add(out, self.bias)
            shape = F.shape(out)
            out = F.reshape(out, (shape[0] * shape[1], shape[2], shape[3]))
            out = self.relu(out)
            return out

    context.set_context(save_graphs=True, save_graphs_path="./test_shape_used_by_one_static")
    dataset_shard = (1, 1, 1, 1)
    successor_shard = (2, 2, 2)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=6, device_num=8,
                                      dataset_strategy=(dataset_shard,))
    model = ShapeUsedByOneOp((1, 2, 1, 1), successor_shard).to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    input_ids = Tensor(np.random.rand(4, 6, 2, 2), dtype=mstype.float16)
    model.set_inputs(input_ids)
    _ = compile_net(model, input_ids)


def test_shape_used_by_two():
    """
    Feature: Accomplish partial shape in dynamic shape scene.
    Description: Test shape used by two successor in dynamic shape.
    Expectation: Compile success.
    """

    class ShapeUsedByTwoOp(nn.Cell):
        def __init__(self, previous_shard, successor_shard):
            super(ShapeUsedByTwoOp, self).__init__()
            bias_shard = (1, 1, 1, 1)
            self.bias = Tensor(np.array([1]).reshape(1, 1, 1, 1))
            self.add = P.Add().shard((previous_shard, bias_shard))
            self.relu = P.ReLU().shard((successor_shard,))
            self.matmul = P.MatMul().shard(((2, 2), (2, 2)))
            self.w = Tensor(np.random.rand(4, 8).astype(np.float16))

        def construct(self, x):
            out = x
            out = self.add(out, self.bias)
            shape = F.shape(out)
            out = F.reshape(out, (shape[0] * shape[1], shape[2], shape[3]))  # 4,6,2,2
            out = self.relu(out)
            out = F.reshape(out, (shape[0] * shape[1], shape[2] * shape[3]))
            out = self.matmul(out, self.w)
            return out

    context.set_context(save_graphs=True, save_graphs_path="./test_shape_used_by_two")
    dataset_shard = (1, 1, 1, 1)
    successor_shard = (2, 2, 2)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8,
                                      dataset_strategy=(dataset_shard,))
    model = ShapeUsedByTwoOp((4, 2, 1, 1), successor_shard).to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    d1 = Symbol(divisor=2)
    input_ids = Tensor(shape=[4, d1, 2, 2], dtype=mstype.float16)
    model.set_inputs(input_ids)
    phase = compile_net(model, input_ids)
    validator = ParallelValidator(model, phase)
    assert validator.check_node_inputs_has('Split-0', ['TupleGetItem-0', 0, 4])
    assert validator.check_node_inputs_has('TupleGetItem-1', ['Split-0', 0])
    assert validator.check_node_inputs_has('Split-1', ['TupleGetItem-1', 1, 2])
    assert validator.check_node_inputs_has('TupleGetItem-2', ['Split-1', 0])
    assert validator.check_node_inputs_has('Add-0', ['TupleGetItem-2', '_GetTensorSlice-0'])
    assert validator.check_node_inputs_has('Shape-0', ['Add-0'])
    assert validator.check_node_inputs_has('tuple_getitem_for_value_6-0', ['Shape-0', 1])
    assert validator.check_node_inputs_has('MakeTuple-1', [1, 1, 'tuple_getitem_for_value_6-0', 2, 2])
    assert validator.check_node_inputs_has('Reshape-0', ['Add-0', 'MakeTuple-1'])
    assert validator.check_node_inputs_has('AllGather-0', ['Reshape-0'])
    assert validator.check_node_inputs_has('Split-2', ['AllGather-0', 0, 2])
    assert validator.check_node_inputs_has('TupleGetItem-3', ['Split-2', 0])
    assert validator.check_node_inputs_has('TupleGetItem-4', ['Split-2', 1])
    assert validator.check_node_inputs_has('MakeTuple-2', ['TupleGetItem-3', 'TupleGetItem-4'])
    assert validator.check_node_inputs_has('Concat-0', ['MakeTuple-2', 2])
    assert validator.check_node_inputs_has('Split-3', ['Concat-0', 4, 2])
    assert validator.check_node_inputs_has('TupleGetItem-5', ['Split-3', 0])
    assert validator.check_node_inputs_has('AllGather-1', ['TupleGetItem-5'])
    assert validator.check_node_inputs_has('Split-4', ['AllGather-1', 0, 2])
    assert validator.check_node_inputs_has('TupleGetItem-6', ['Split-4', 0])
    assert validator.check_node_inputs_has('TupleGetItem-7', ['Split-4', 1])
    assert validator.check_node_inputs_has('MakeTuple-3', ['TupleGetItem-6', 'TupleGetItem-7'])
    assert validator.check_node_inputs_has('Concat-1', ['MakeTuple-3', 1])
    assert validator.check_node_inputs_has('Split-5', ['Concat-1', 3, 2])
    assert validator.check_node_inputs_has('TupleGetItem-8', ['Split-5', 0])
    assert validator.check_node_inputs_fuzzy_match('Reshape-1', ['TupleGetItem-8', '(-1, 1, 1)'])
    assert validator.check_node_inputs_has('ReLU-0', ['Reshape-1'])
    assert validator.check_node_inputs_has('AllGather-2', ['ReLU-0'])
    assert validator.check_node_inputs_has('Split-6', ['AllGather-2', 0, 2])
    assert validator.check_node_inputs_has('TupleGetItem-9', ['Split-6', 0])
    assert validator.check_node_inputs_has('TupleGetItem-10', ['Split-6', 1])
    assert validator.check_node_inputs_has('MakeTuple-4', ['TupleGetItem-9', 'TupleGetItem-10'])
    assert validator.check_node_inputs_has('Concat-2', ['MakeTuple-4', 2])
    assert validator.check_node_inputs_fuzzy_match('Reshape-2', ['Concat-2', '(-1, 2)'])
    assert validator.check_node_inputs_has('MatMul-0', ['Reshape-2', '_GetTensorSlice-1'])
    assert validator.check_node_inputs_has('AllReduce-0', ['MatMul-0'])


def test_shape_used_by_two_static():
    """
    Feature: Accomplish partial shape in dynamic shape scene.
    Description: Test shape used by two successor in dynamic shape.
    Expectation: Compile success.
    """

    class ShapeUsedByTwoOp(nn.Cell):
        def __init__(self, previous_shard, successor_shard):
            super(ShapeUsedByTwoOp, self).__init__()
            bias_shard = (1, 1, 1, 1)
            self.bias = Tensor(np.array([1]).reshape(1, 1, 1, 1))
            self.add = P.Add().shard((previous_shard, bias_shard))
            self.relu = P.ReLU().shard((successor_shard,))
            self.matmul = P.MatMul().shard(((2, 2), (2, 2)))
            self.w = Tensor(np.random.rand(4, 8).astype(np.float16))

        def construct(self, x):
            out = x
            out = self.add(out, self.bias)
            shape = F.shape(out)
            out = F.reshape(out, (shape[0] * shape[1], shape[2], shape[3]))
            out = self.relu(out)
            out = F.reshape(out, (shape[0] * shape[1], shape[2] * shape[3]))
            out = self.matmul(out, self.w)
            return out

    context.set_context(save_graphs=True, save_graphs_path="./test_shape_used_by_two_static")
    dataset_shard = (1, 1, 1, 1)
    successor_shard = (2, 2, 2)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8,
                                      dataset_strategy=(dataset_shard,))
    model = ShapeUsedByTwoOp((4, 2, 1, 1), successor_shard).to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    input_ids = Tensor(np.random.rand(4, 6, 2, 2), dtype=mstype.float16)
    model.set_inputs(input_ids)
    _ = compile_net(model, input_ids)


def test_two_dynamic_dims_used_by_two():
    """
    Feature: Accomplish partial shape in dynamic shape scene.
    Description: Test shape used by two successor in dynamic shape.
    Expectation: Compile success.
    """

    class ShapeUsedByTwoOp(nn.Cell):
        def __init__(self, previous_shard, successor_shard):
            super(ShapeUsedByTwoOp, self).__init__()
            bias_shard = (1, 1, 1, 1)
            self.bias = Tensor(np.array([1]).reshape(1, 1, 1, 1))
            self.add = P.Add().shard((previous_shard, bias_shard))
            self.relu = P.ReLU().shard((successor_shard,))
            self.matmul = P.MatMul().shard(((2, 2), (2, 2)))
            self.w = Tensor(np.random.rand(4, 8).astype(np.float16))

        def construct(self, x):
            out = x  # B,N,D,H
            out = self.add(out, self.bias)  # shard (4, 2, 1, 1)
            shape = F.shape(out)
            out = F.reshape(out, (shape[0] * shape[1], shape[2], shape[3]))  # B*N,D,H
            out = self.relu(out)  # shard (2, 2, 2)
            out = F.reshape(out, (shape[0] * shape[1], shape[2] * shape[3]))  # B*N,D*H
            out = self.matmul(out, self.w)
            return out

    context.set_context(save_graphs=True, save_graphs_path="./test_two_dynamic_dims_used_by_two")
    dataset_shard = (1, 1, 1, 1)
    successor_shard = (2, 2, 2)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8,
                                      dataset_strategy=(dataset_shard,))
    model = ShapeUsedByTwoOp((4, 2, 1, 1), successor_shard).to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    d0 = Symbol(divisor=4)
    d1 = Symbol(divisor=2)
    input_ids = Tensor(shape=[d0, d1, 2, 2], dtype=mstype.float16)
    model.set_inputs(input_ids)
    phase = compile_net(model, input_ids)
    validator = ParallelValidator(model, phase)
    assert validator.check_node_inputs_has('TupleGetItem-0', ['MakeTuple-0', 0])
    assert validator.check_node_inputs_has('Split-0', ['TupleGetItem-0', 0, 4])
    assert validator.check_node_inputs_has('TupleGetItem-1', ['Split-0', 0])
    assert validator.check_node_inputs_has('Split-1', ['TupleGetItem-1', 1, 2])
    assert validator.check_node_inputs_has('TupleGetItem-2', ['Split-1', 0])
    assert validator.check_node_inputs_has('Add-0', ['TupleGetItem-2', '_GetTensorSlice-0'])
    assert validator.check_node_inputs_has('Shape-0', ['Add-0'])
    assert validator.check_node_inputs_has('tuple_getitem_for_value_12-0', ['Shape-0', 0])
    assert validator.check_node_inputs_has('tuple_getitem_for_value_10-0', ['Shape-0', 1])
    assert validator.check_node_inputs_has('MakeTuple-1',
                                           [1, 'tuple_getitem_for_value_12-0', 'tuple_getitem_for_value_10-0', 2, 2])
    assert validator.check_node_inputs_has('Reshape-0', ['Add-0', 'MakeTuple-1'])
    assert validator.check_node_inputs_has('AllGather-0', ['Reshape-0'])
    assert validator.check_node_inputs_has('Split-2', ['AllGather-0', 0, 2])
    assert validator.check_node_inputs_has('TupleGetItem-3', ['Split-2', 0])
    assert validator.check_node_inputs_has('TupleGetItem-4', ['Split-2', 1])
    assert validator.check_node_inputs_has('MakeTuple-2', ['TupleGetItem-3', 'TupleGetItem-4'])
    assert validator.check_node_inputs_has('Concat-0', ['MakeTuple-2', 2])
    assert validator.check_node_inputs_has('Split-3', ['Concat-0', 4, 2])
    assert validator.check_node_inputs_has('TupleGetItem-5', ['Split-3', 0])
    assert validator.check_node_inputs_has('AllGather-1', ['TupleGetItem-5'])
    assert validator.check_node_inputs_has('Split-4', ['AllGather-1', 0, 2])
    assert validator.check_node_inputs_has('TupleGetItem-6', ['Split-4', 0])
    assert validator.check_node_inputs_has('TupleGetItem-7', ['Split-4', 1])
    assert validator.check_node_inputs_has('MakeTuple-3', ['TupleGetItem-6', 'TupleGetItem-7'])
    assert validator.check_node_inputs_has('Concat-1', ['MakeTuple-3', 1])
    assert validator.check_node_inputs_has('Split-5', ['Concat-1', 3, 2])
    assert validator.check_node_inputs_has('TupleGetItem-8', ['Split-5', 0])
    assert validator.check_node_inputs_fuzzy_match('Reshape-1', ['TupleGetItem-8', '(-1, 1, 1)'])
    assert validator.check_node_inputs_has('ReLU-0', ['Reshape-1'])
    assert validator.check_node_inputs_has('AllGather-2', ['ReLU-0'])
    assert validator.check_node_inputs_has('Split-6', ['AllGather-2', 0, 2])
    assert validator.check_node_inputs_has('TupleGetItem-9', ['Split-6', 0])
    assert validator.check_node_inputs_has('TupleGetItem-10', ['Split-6', 1])
    assert validator.check_node_inputs_has('MakeTuple-4', ['TupleGetItem-9', 'TupleGetItem-10'])
    assert validator.check_node_inputs_has('Concat-2', ['MakeTuple-4', 2])
    assert validator.check_node_inputs_fuzzy_match('Reshape-2', ['Concat-2', '(-1, 2)'])
    assert validator.check_node_inputs_has('MatMul-0', ['Reshape-2', '_GetTensorSlice-1'])
    assert validator.check_node_inputs_has('AllReduce-0', ['MatMul-0'])


def test_two_dynamic_dims_used_by_two_static():
    """
    Feature: Accomplish partial shape in dynamic shape scene.
    Description: Test shape used by two successor in dynamic shape.
    Expectation: Compile success.
    """

    class ShapeUsedByTwoOp(nn.Cell):
        def __init__(self, previous_shard, successor_shard):
            super(ShapeUsedByTwoOp, self).__init__()
            bias_shard = (1, 1, 1, 1)
            self.bias = Tensor(np.array([1]).reshape(1, 1, 1, 1))
            self.add = P.Add().shard((previous_shard, bias_shard))
            self.relu = P.ReLU().shard((successor_shard,))
            self.matmul = P.MatMul().shard(((2, 2), (2, 2)))
            self.w = Tensor(np.random.rand(4, 8).astype(np.float16))

        def construct(self, x):
            out = x  # B,N,D,H
            out = self.add(out, self.bias)  # shard (4, 2, 1, 1)
            shape = F.shape(out)
            out = F.reshape(out, (shape[0] * shape[1], shape[2], shape[3]))  # B*N,D,H
            out = self.relu(out)  # shard (2, 2, 2)
            out = F.reshape(out, (shape[0] * shape[1], shape[2] * shape[3]))  # B*N,D*H
            out = self.matmul(out, self.w)
            return out

    context.set_context(save_graphs=True, save_graphs_path="./test_two_dynamic_dims_used_by_two_static")
    dataset_shard = (1, 1, 1, 1)
    successor_shard = (2, 2, 2)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8,
                                      dataset_strategy=(dataset_shard,))
    model = ShapeUsedByTwoOp((4, 2, 1, 1), successor_shard).to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    input_ids = Tensor(np.random.rand(12, 10, 2, 2), dtype=mstype.float16)
    model.set_inputs(input_ids)
    _ = compile_net(model, input_ids)


def test_not_equal_cast_mul_need_allgather_static():
    """
    Feature: Test transformer sub-structure NotEqual-Cast-Mul with allgather.
    Description: Test transformer sub-structure NotEqual-Cast-Mul with allgather.
    Expectation: Compile success.
    """

    class SubNet(nn.Cell):
        def __init__(self):
            super(SubNet, self).__init__()
            self.mask = Tensor([1], dtype=mstype.float16)
            self.not_equal = P.NotEqual().shard(((1, 2), (1,)))
            self.mul = P.Mul().shard(((1,), (1,)))
            self.cast = P.Cast()
            self.reshape = P.Reshape()

        def construct(self, x):
            out = x
            out = self.not_equal(out, self.mask)
            out = self.cast(out, mstype.float16)
            out = self.reshape(out, (-1,))
            out = self.mul(out, out)
            return out

    context.set_context(save_graphs=True, save_graphs_path="./test_not_equal_cast_mul_need_allgather_static")
    dataset_shard = (1, 1)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8,
                                      dataset_strategy=(dataset_shard,))
    model = SubNet().to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    input_ids = Tensor(np.random.rand(4, 6), dtype=mstype.float16)
    model.set_inputs(input_ids)
    _ = compile_net(model, input_ids)


def test_not_equal_cast_mul_need_allgather_dyn():
    """
    Feature: Test transformer sub-structure NotEqual-Cast-Mul with allgather.
    Description: Test transformer sub-structure NotEqual-Cast-Mul with allgather.
    Expectation: Compile success.
    """

    class SubNet(nn.Cell):
        def __init__(self):
            super(SubNet, self).__init__()
            self.mask = Tensor([1], dtype=mstype.float16)
            self.not_equal = P.NotEqual().shard(((1, 2), (1,)))
            self.mul = P.Mul().shard(((1,), (1,)))
            self.cast = P.Cast()
            self.reshape = P.Reshape()

        def construct(self, x):
            out = x
            out = self.not_equal(out, self.mask)
            out = self.cast(out, mstype.float16)
            out = self.reshape(out, (-1,))
            out = self.mul(out, out)
            return out

    context.set_context(save_graphs=True, save_graphs_path="./test_not_equal_cast_mul_need_allgather_dyn")
    dataset_shard = (1, 1)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8,
                                      dataset_strategy=(dataset_shard,))
    model = SubNet().to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    d1 = Symbol(divisor=2)
    input_ids = Tensor(shape=[4, d1], dtype=mstype.float16)
    model.set_inputs(input_ids)
    phase = compile_net(model, input_ids)
    validator = ParallelValidator(model, phase)
    assert validator.check_node_inputs_has('TupleGetItem-0', ['MakeTuple-0', 0])
    assert validator.check_node_inputs_has('Split-0', ['TupleGetItem-0', 1, 2])
    assert validator.check_node_inputs_has('TupleGetItem-1', ['Split-0', 0])
    assert validator.check_node_inputs_fuzzy_match('NotEqual-0', ['TupleGetItem-1', '1.0'])
    assert validator.check_node_inputs_has('Cast-0', ['NotEqual-0', 42])
    assert validator.check_node_inputs_has('AllGather-0', ['Cast-0'])
    assert validator.check_node_inputs_has('Split-1', ['AllGather-0', 0, 2])
    assert validator.check_node_inputs_has('TupleGetItem-2', ['Split-1', 0])
    assert validator.check_node_inputs_has('TupleGetItem-3', ['Split-1', 1])
    assert validator.check_node_inputs_has('MakeTuple-1', ['TupleGetItem-2', 'TupleGetItem-3'])
    assert validator.check_node_inputs_has('Concat-0', ['MakeTuple-1', 1])
    assert validator.check_node_inputs_fuzzy_match('Reshape-0', ['Concat-0', '(-1)'])
    assert validator.check_node_inputs_has('Mul-0', ['Reshape-0', 'Reshape-0'])


def test_not_equal_cast_mul_no_need_allgather_static():
    """
    Feature: Test transformer sub-structure NotEqual-Cast-Mul without allgather.
    Description: Test transformer sub-structure NotEqual-Cast-Mul without allgather.
    Expectation: Compile success.
    """

    class SubNet(nn.Cell):
        def __init__(self):
            super(SubNet, self).__init__()
            self.mask = Tensor([1], dtype=mstype.float16)
            self.not_equal = P.NotEqual().shard(((2, 1), (1,)))
            self.mul = P.Mul().shard(((2,), (2,)))
            self.cast = P.Cast()
            self.reshape = P.Reshape()

        def construct(self, x):
            out = x
            out = self.not_equal(out, self.mask)
            out = self.cast(out, mstype.float16)
            out = self.reshape(out, (-1,))
            out = self.mul(out, out)
            return out

    context.set_context(save_graphs=True, save_graphs_path="./test_not_equal_cast_mul_no_need_allgather_static")
    dataset_shard = (1, 1)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8,
                                      dataset_strategy=(dataset_shard,))
    model = SubNet().to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    input_ids = Tensor(np.random.rand(2, 6), dtype=mstype.float16)
    model.set_inputs(input_ids)
    _ = compile_net(model, input_ids)


def test_not_equal_cast_mul_no_need_allgather_dyn():
    """
    Feature: Test transformer sub-structure NotEqual-Cast-Mul without allgather.
    Description: Test transformer sub-structure NotEqual-Cast-Mul without allgather.
    Expectation: Compile success.
    """

    class SubNet(nn.Cell):
        def __init__(self):
            super(SubNet, self).__init__()
            self.mask = Tensor([1], dtype=mstype.float16)
            self.not_equal = P.NotEqual().shard(((2, 1), (1,)))
            self.mul = P.Mul().shard(((2,), (2,)))
            self.cast = P.Cast()
            self.reshape = P.Reshape()

        def construct(self, x):
            out = x
            out = self.not_equal(out, self.mask)
            out = self.cast(out, mstype.float16)
            out = self.reshape(out, (-1,))
            out = self.mul(out, out)
            return out

    context.set_context(save_graphs=True, save_graphs_path="./test_not_equal_cast_mul_no_need_allgather_dyn")
    dataset_shard = (1, 1)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8,
                                      dataset_strategy=(dataset_shard,))
    model = SubNet().to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    d1 = Symbol(divisor=2)
    input_ids = Tensor(shape=[2, d1], dtype=mstype.float16)
    model.set_inputs(input_ids)
    phase = compile_net(model, input_ids)
    validator = ParallelValidator(model, phase)
    assert validator.check_node_inputs_has('MakeTuple-0', ['inputs0'])
    assert validator.check_node_inputs_has('TupleGetItem-0', ['MakeTuple-0', 0])
    assert validator.check_node_inputs_has('Split-0', ['TupleGetItem-0', 0, 2])
    assert validator.check_node_inputs_has('TupleGetItem-1', ['Split-0', 0])
    assert validator.check_node_inputs_has('NotEqual-0', ['TupleGetItem-1', '1.0'])
    assert validator.check_node_inputs_has('Cast-0', ['NotEqual-0', 42])
    assert validator.check_node_inputs_has('Reshape-0', ['Cast-0', '(-1)'])
    assert validator.check_node_inputs_has('Mul-0', ['Reshape-0', 'Reshape-0'])


def test_reshape_shape_overflow():
    """
    Feature: Test transformer sub-structure Mul-Reshape-MatMul.
    Description: Test reshape shape value overflow.
    Expectation: Compile success and assertion passed.
    """

    class SubNet(nn.Cell):
        def __init__(self):
            super(SubNet, self).__init__()
            self.bias = Tensor(np.array([1] * 4096).reshape(4096), dtype=mstype.float16)
            self.mul = P.Mul().shard(((2, 4, 1), (1,)))
            self.matmul = P.MatMul(transpose_b=True).shard(((2, 1), (4, 1)))
            self.w = Tensor(np.random.rand(4096, 4096).astype(np.float16))

        def construct(self, x):
            out = x
            out = self.mul(out, self.bias)  # shape: (2, -1, 4096), shard: (2,4,1)
            out = F.reshape(out, (-1, 4096))
            out = self.matmul(out, self.w)  # shard: ((2, 1), (4, 1))
            return out

    context.set_context(save_graphs=True, save_graphs_path="./test_reshape_shape_overflow")
    dataset_shard = (1, 1, 1)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8,
                                      dataset_strategy=(dataset_shard,))
    model = SubNet().to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    d1 = Symbol(divisor=4)
    input_ids = Tensor(shape=[2, d1, 4096], dtype=mstype.float16)
    model.set_inputs(input_ids)
    phase = compile_net(model, input_ids)
    validator = ParallelValidator(model, phase)
    assert validator.check_node_inputs_fuzzy_match('Reshape-0', ["", "-1, 4096"])


def test_dataset_shard_is_same_with_first_op():
    """
    Feature: Test redistribution input is indirect association to VirtualDataset.
    Description: Test redistribution input is indirect association to VirtualDataset.
    Expectation: Compile success and assertion passed.
    """

    class ParallelArgMaxWithValueNet(nn.Cell):
        def __init__(self, mul_size, mul2_size, keep_dims=False, axis=-1,
                     strategy=None, strategy2=None):
            super(ParallelArgMaxWithValueNet, self).__init__()
            mul_np = np.full(mul_size, 0.5, dtype=np.float32)
            mul2_np = np.full(mul2_size, 0.5, dtype=np.float32)
            self.mul_weight = Parameter(Tensor(mul_np), name="mul_weight")
            self.mul2_weight = Parameter(Tensor(mul2_np), name="mul2_weight")
            self.mul = P.Mul()
            self.mul2 = P.Mul()
            self.arg_max_with_value = P.ArgMaxWithValue(keep_dims=keep_dims, axis=axis)
            if strategy is not None and strategy2 is not None:
                self.arg_max_with_value.shard(strategy)
                self.mul.shard(strategy2)
                if keep_dims:
                    self.mul2.shard(((1, 1), (1, 1)))
                else:
                    self.mul2.shard(((1,), (1,)))

        def construct(self, inputs):
            x = self.mul(inputs, self.mul_weight)  # shard: (4, 1), (1, 1)
            x = self.arg_max_with_value(x)[1]  # shard: (4, 1)
            x = self.mul2(x, self.mul2_weight)
            return x

    context.set_context(save_graphs=True, save_graphs_path="./test_dataset_shard_is_same_with_first_op")

    dataset_shard = (1, 1)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8,
                                      dataset_strategy=(dataset_shard,))
    model = ParallelArgMaxWithValueNet(mul_size=(1, 1), mul2_size=(128, 1), keep_dims=True,
                                       strategy=((4, 1),), strategy2=((4, 1), (1, 1))).to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    d0 = Symbol(divisor=8)
    x = Tensor(shape=[d0, None], dtype=mstype.float32)
    model.set_inputs(x)
    phase = compile_net(model, x)
    validator = ParallelValidator(model, phase)
    assert validator.check_node_inputs_has('TupleGetItem-0', ['MakeTuple-0', 0])
    assert validator.check_node_inputs_has('Cast-0', ['TupleGetItem-0', 42])
    assert validator.check_node_inputs_has('Split-0', ['Cast-0', 0, 4])
    assert validator.check_node_inputs_has('TupleGetItem-1', ['Split-0', 0])
    assert validator.check_node_inputs_has('Cast-1', ['Load-0', 42])
    assert validator.check_node_inputs_has('Mul-0', ['TupleGetItem-1', 'Cast-1'])
    assert validator.check_node_inputs_has('ArgMaxWithValue-0', ['Mul-0', -1, True])
    assert validator.check_node_inputs_has('TupleGetItem-2', ['ArgMaxWithValue-0', 1])
    assert validator.check_node_inputs_has('Shape-0', ['TupleGetItem-2'])
    assert validator.check_node_inputs_has('tuple_getitem_for_value_12-0', ['Shape-0', 0])
    assert validator.check_node_inputs_has('MakeTuple-1', ['tuple_getitem_for_value_12-0'])
    assert validator.check_node_inputs_has('Reshape-0', ['TupleGetItem-2', 'MakeTuple-1'])
    assert validator.check_node_inputs_has('AllGather-0', ['Reshape-0'])
    assert validator.check_node_inputs_has('Reshape-1', ['AllGather-0', '(-1, 1)'])
    assert validator.check_node_inputs_has('Cast-2', ['Load-1', 42])
    assert validator.check_node_inputs_has('Mul-1', ['Reshape-1', 'Cast-2'])
    assert validator.check_node_inputs_has('MakeTuple-2', ['Load-1', 'Load-0'])


def test_dataset_shard_is_not_same_with_first_op():
    """
    Feature: Test redistribution input is indirect association to VirtualDataset.
    Description: Test redistribution input is indirect association to VirtualDataset.
    Expectation: Compile success and assertion passed.
    """

    class ParallelArgMaxWithValueNet(nn.Cell):
        def __init__(self, mul_size, mul2_size, keep_dims=False, axis=-1,
                     strategy=None, strategy2=None):
            super(ParallelArgMaxWithValueNet, self).__init__()
            mul_np = np.full(mul_size, 0.5, dtype=np.float32)
            mul2_np = np.full(mul2_size, 0.5, dtype=np.float32)
            self.mul_weight = Parameter(Tensor(mul_np), name="mul_weight")
            self.mul2_weight = Parameter(Tensor(mul2_np), name="mul2_weight")
            self.mul = P.Mul()
            self.mul2 = P.Mul()
            self.arg_max_with_value = P.ArgMaxWithValue(keep_dims=keep_dims, axis=axis)
            if strategy is not None and strategy2 is not None:
                self.arg_max_with_value.shard(strategy)
                self.mul.shard(strategy2)
                if keep_dims:
                    self.mul2.shard(((1, 1), (1, 1)))
                else:
                    self.mul2.shard(((1,), (1,)))

        def construct(self, inputs):
            x = self.mul(inputs, self.mul_weight)  # shard: (4, 1), (1, 1)
            x = self.arg_max_with_value(x)[1]  # shard: (4, 1)
            x = self.mul2(x, self.mul2_weight)
            return x

    context.set_context(save_graphs=True, save_graphs_path="./test_dataset_shard_is_not_same_with_first_op")

    dataset_shard = (8, 1)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8,
                                      dataset_strategy=(dataset_shard,))
    model = ParallelArgMaxWithValueNet(mul_size=(1, 1), mul2_size=(128, 1), keep_dims=True,
                                       strategy=((4, 1),), strategy2=((4, 1), (1, 1))).to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    d0 = Symbol(divisor=8)
    x = Tensor(shape=[d0, 4], dtype=mstype.float32)
    model.set_inputs(x)
    phase = compile_net(model, x)
    validator = ParallelValidator(model, phase)
    assert validator.check_node_inputs_has('TupleGetItem-0', ['MakeTuple-0', 0])
    assert validator.check_node_inputs_has('Cast-0', ['TupleGetItem-0', 42])
    assert validator.check_node_inputs_has('Shape-0', ['Cast-0'])
    assert validator.check_node_inputs_has('tuple_getitem_for_value_24-0', ['Shape-0', 0])
    assert validator.check_node_inputs_has('MakeTuple-1', [1, 'tuple_getitem_for_value_24-0', 4])
    assert validator.check_node_inputs_has('Reshape-0', ['Cast-0', 'MakeTuple-1'])
    assert validator.check_node_inputs_has('AllGather-0', ['Reshape-0'])
    assert validator.check_node_inputs_has('Split-0', ['AllGather-0', 0, 2])
    assert validator.check_node_inputs_has('TupleGetItem-1', ['Split-0', 0])
    assert validator.check_node_inputs_has('TupleGetItem-2', ['Split-0', 1])
    assert validator.check_node_inputs_has('MakeTuple-2', ['TupleGetItem-1', 'TupleGetItem-2'])
    assert validator.check_node_inputs_has('Concat-0', ['MakeTuple-2', 1])
    assert validator.check_node_inputs_has('Reshape-1', ['Concat-0', '(-1, 4)'])
    assert validator.check_node_inputs_has('Cast-1', ['Load-0', 42])
    assert validator.check_node_inputs_has('Mul-0', ['Reshape-1', 'Cast-1'])
    assert validator.check_node_inputs_has('ArgMaxWithValue-0', ['Mul-0', -1, True])
    assert validator.check_node_inputs_has('TupleGetItem-3', ['ArgMaxWithValue-0', 1])
    assert validator.check_node_inputs_has('Shape-1', ['TupleGetItem-3'])
    assert validator.check_node_inputs_has('tuple_getitem_for_value_12-0', ['Shape-1', 0])
    assert validator.check_node_inputs_has('MakeTuple-3', ['tuple_getitem_for_value_12-0'])
    assert validator.check_node_inputs_has('Reshape-2', ['TupleGetItem-3', 'MakeTuple-3'])
    assert validator.check_node_inputs_has('AllGather-1', ['Reshape-2'])
    assert validator.check_node_inputs_has('Reshape-3', ['AllGather-1', '(-1, 1)'])
    assert validator.check_node_inputs_has('Mul-1', ['Reshape-3', 'Cast-2'])


def test_greater_equal_op_with_two_dynamic_axis():
    """
    Feature: Test tensor redistribution with GreaterEqual op.
    Description: Test GreaterEqual op which input has two dynamic axis.
    Expectation: Compile success and assertion passed.
    """

    class ParallelGreaterEqualFlattenDivNet(nn.Cell):
        def __init__(self):
            super(ParallelGreaterEqualFlattenDivNet, self).__init__()
            weight_np = np.random.randn(*(8, 4)).astype(np.float32)
            self.div_weight = Parameter(Tensor(weight_np), name="div_weight")
            self.flat = nn.Flatten()
            self.div = P.Div()
            self.cast = P.Cast()
            self.greaterequal = P.GreaterEqual()
            self.greaterequal.shard(((4, 1), (4, 1)))

        def construct(self, inputs):
            x = self.greaterequal(inputs, inputs)
            x = self.cast(x, mstype.float16)
            x = self.flat(x)
            x = self.div(x, self.flat(inputs))
            return x

    context.set_context(save_graphs=True, save_graphs_path="./test_greater_equal_op")
    dataset_shard = (1, 1)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8,
                                      dataset_strategy=(dataset_shard,))
    model = ParallelGreaterEqualFlattenDivNet()
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    s1 = Symbol(divisor=8)
    s2 = Symbol(divisor=1)
    x = Tensor(shape=[s1, s2], dtype=mstype.float32)
    model.set_inputs(x)
    phase = compile_net(model, x)
    validator = ParallelValidator(model, phase)
    assert validator.check_node_inputs_has('MakeTuple-0', ['inputs0'])
    assert validator.check_node_inputs_has('TupleGetItem-0', ['MakeTuple-0', 0])
    assert validator.check_node_inputs_has('Split-0', ['TupleGetItem-0', 0, 4])
    assert validator.check_node_inputs_has('TupleGetItem-1', ['Split-0', 0])
    assert validator.check_node_inputs_has('Split-1', ['TupleGetItem-0', 0, 4])
    assert validator.check_node_inputs_has('TupleGetItem-2', ['Split-1', 0])
    assert validator.check_node_inputs_has('GreaterEqual-0', ['TupleGetItem-1', 'TupleGetItem-2'])
    assert validator.check_node_inputs_has('Cast-0', ['GreaterEqual-0', 42])
    assert validator.check_node_inputs_has('AllGather-0', ['Cast-0'])
    assert validator.check_node_inputs_has('Flatten-0', ['AllGather-0'])
    assert validator.check_node_inputs_has('Cast-1', ['Flatten-0', 43])
    assert validator.check_node_inputs_has('Flatten-1', ['TupleGetItem-0'])
    assert validator.check_node_inputs_has('Div-0', ['Cast-1', 'Flatten-1'])


def test_two_matmul_with_different_layout():
    """
    Feature: Test tensor redistribution in dynamic shape.
    Description: Test insert redistribution between two matmul op.
    Expectation: Compile success and assertion passed.
    """

    class MatMulNet(nn.Cell):
        def __init__(self):
            super(MatMulNet, self).__init__()
            self.matmul1 = P.MatMul()
            self.matmul2 = P.MatMul()
            self.matmul1_weight = Parameter(np.full((64, 32), 0.5, dtype=np.float32), name="weight1")
            self.matmul2_weight = Parameter(np.full((32, 32), 0.8, dtype=np.float32), name="weight2")
            self.matmul1.shard(((4, 1), (1, 2)))
            self.matmul2.shard(((2, 2), (2, 2)))
            self.relu = nn.ReLU()

        def construct(self, x):
            x = self.matmul1(x, self.matmul1_weight)
            x = self.matmul2(x, self.matmul2_weight)
            x = self.relu(x)
            return x

    context.set_context(save_graphs=True, save_graphs_path="./test_two_matmul_with_different_layout")
    dataset_shard = (1, 1)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8,
                                      dataset_strategy=(dataset_shard,))
    model = MatMulNet()
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    d0 = Symbol(divisor=8)
    x = Tensor(shape=[d0, 64], dtype=mstype.float32)
    model.set_inputs(x)
    phase = compile_net(model, x)
    validator = ParallelValidator(model, phase)
    assert validator.check_node_inputs_has('TupleGetItem-0', ['MakeTuple-0', 0])
    assert validator.check_node_inputs_has('Split-0', ['TupleGetItem-0', 0, 4])
    assert validator.check_node_inputs_has('TupleGetItem-1', ['Split-0', 0])
    assert validator.check_node_inputs_has('MatMul-0', ['TupleGetItem-1', 'Load-0'])
    assert validator.check_node_inputs_has('Shape-0', ['MatMul-0'])
    assert validator.check_node_inputs_has('tuple_getitem_for_value_12-0', ['Shape-0', 0])
    assert validator.check_node_inputs_has('MakeTuple-1', [1, 'tuple_getitem_for_value_12-0', 16])
    assert validator.check_node_inputs_has('Reshape-0', ['MatMul-0', 'MakeTuple-1'])
    assert validator.check_node_inputs_has('AllGather-0', ['Reshape-0'])
    assert validator.check_node_inputs_has('Split-1', ['AllGather-0', 0, 2])
    assert validator.check_node_inputs_has('TupleGetItem-2', ['Split-1', 0])
    assert validator.check_node_inputs_has('TupleGetItem-3', ['Split-1', 1])
    assert validator.check_node_inputs_has('MakeTuple-2', ['TupleGetItem-2', 'TupleGetItem-3'])
    assert validator.check_node_inputs_has('Concat-0', ['MakeTuple-2', 2])
    assert validator.check_node_inputs_has('AllGather-1', ['Concat-0'])
    assert validator.check_node_inputs_has('Split-2', ['AllGather-1', 0, 2])
    assert validator.check_node_inputs_has('TupleGetItem-4', ['Split-2', 0])
    assert validator.check_node_inputs_has('TupleGetItem-5', ['Split-2', 1])
    assert validator.check_node_inputs_has('MakeTuple-3', ['TupleGetItem-4', 'TupleGetItem-5'])
    assert validator.check_node_inputs_has('Concat-1', ['MakeTuple-3', 1])
    assert validator.check_node_inputs_has('Split-3', ['Concat-1', 2, 2])
    assert validator.check_node_inputs_has('TupleGetItem-6', ['Split-3', 0])
    assert validator.check_node_inputs_has('Reshape-1', ['TupleGetItem-6', '(-1, 16)'])
    assert validator.check_node_inputs_has('MatMul-1', ['Reshape-1', 'Load-1'])
    assert validator.check_node_inputs_has('AllReduce-0', ['MatMul-1'])
    assert validator.check_node_inputs_has('Shape-1', ['MatMul-1'])
    assert validator.check_node_inputs_has('tuple_getitem_for_value_48-0', ['Shape-1', 0])
    assert validator.check_node_inputs_has('ScalarDiv-0', ['tuple_getitem_for_value_48-0', 12])
    assert validator.check_node_inputs_has('ScalarCast-0', ['ScalarDiv-0', 35])
    assert validator.check_node_inputs_has('ScalarDiv-1', ['tuple_getitem_for_value_48-0', 2])
    assert validator.check_node_inputs_has('ScalarCast-1', ['ScalarDiv-1', 35])
    assert validator.check_node_inputs_has('MakeTuple-4', [1, 'ScalarCast-0', 'ScalarCast-1', 16])
    assert validator.check_node_inputs_has('Reshape-2', ['AllReduce-0', 'MakeTuple-4'])
    assert validator.check_node_inputs_has('Split-4', ['Reshape-2', 1, 2])
    assert validator.check_node_inputs_has('TupleGetItem-7', ['Split-4', 0])
    assert validator.check_node_inputs_has('AllGather-2', ['TupleGetItem-7'])
    assert validator.check_node_inputs_has('Split-5', ['AllGather-2', 0, 2])
    assert validator.check_node_inputs_has('TupleGetItem-8', ['Split-5', 0])
    assert validator.check_node_inputs_has('TupleGetItem-9', ['Split-5', 1])
    assert validator.check_node_inputs_has('MakeTuple-5', ['TupleGetItem-8', 'TupleGetItem-9'])
    assert validator.check_node_inputs_has('Concat-2', ['MakeTuple-5', 3])
    assert validator.check_node_inputs_has('Split-6', ['Concat-2', 2, 2])
    assert validator.check_node_inputs_has('TupleGetItem-10', ['Split-6', 0])
    assert validator.check_node_inputs_has('Reshape-3', ['TupleGetItem-10', '(-1, 32)'])
    assert validator.check_node_inputs_has('ReLU-0', ['Reshape-3'])


def test_shrink_four_dims_into_two_dims():
    """
    Feature: Test tensor redistribution in dynamic shape.
    Description: Test shrink four dims into two dims.
    Expectation: Compile success and assertion passed.
    """

    class ReshapeNet(nn.Cell):
        def __init__(self):
            super(ReshapeNet, self).__init__()
            self.add = P.Add().shard(((2, 2, 2, 1), (1, 1, 1, 1)))
            self.relu = P.ReLU().shard(((4, 1),))
            self.weight = Parameter(np.full((1, 1, 1, 1), 0.5, dtype=np.float32), name="weight")
            self.reshape = P.Reshape()
            self.shape = P.Shape()

        def construct(self, x):
            x = self.add(x, self.weight)
            shape = self.shape(x)
            x = self.reshape(x, (shape[0] * shape[1], shape[2] * shape[3]))
            x = self.relu(x)
            return x

    context.set_context(save_graphs=True, save_graphs_path="./test_shrink_four_dims_into_two_dims")
    dataset_shard = (1, 1, 1, 1)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8,
                                      dataset_strategy=(dataset_shard,))
    model = ReshapeNet()
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    d0 = Symbol(divisor=8)
    x = Tensor(shape=[d0, 32, 16, 16], dtype=mstype.float32)
    model.set_inputs(x)
    phase = compile_net(model, x)
    validator = ParallelValidator(model, phase)
    assert validator.check_node_inputs_has('TupleGetItem-0', ['MakeTuple-0', 0])
    assert validator.check_node_inputs_has('Split-0', ['TupleGetItem-0', 0, 2])
    assert validator.check_node_inputs_has('TupleGetItem-1', ['Split-0', 0])
    assert validator.check_node_inputs_has('Split-1', ['TupleGetItem-1', 1, 2])
    assert validator.check_node_inputs_has('TupleGetItem-2', ['Split-1', 0])
    assert validator.check_node_inputs_has('Split-2', ['TupleGetItem-2', 2, 2])
    assert validator.check_node_inputs_has('TupleGetItem-3', ['Split-2', 0])
    assert validator.check_node_inputs_has('Add-0', ['TupleGetItem-3', 'Load-0'])
    assert validator.check_node_inputs_has('AllGather-0', ['Add-0'])
    assert validator.check_node_inputs_has('AllGather-1', ['AllGather-0'])
    assert validator.check_node_inputs_has('Split-3', ['AllGather-1', 0, 2])
    assert validator.check_node_inputs_has('TupleGetItem-4', ['Split-3', 0])
    assert validator.check_node_inputs_has('TupleGetItem-5', ['Split-3', 1])
    assert validator.check_node_inputs_has('MakeTuple-1', ['TupleGetItem-4', 'TupleGetItem-5'])
    assert validator.check_node_inputs_has('Concat-0', ['MakeTuple-1', 1])
    assert validator.check_node_inputs_has('AllGather-2', ['Concat-0'])
    assert validator.check_node_inputs_has('Split-4', ['AllGather-2', 0, 2])
    assert validator.check_node_inputs_has('TupleGetItem-6', ['Split-4', 0])
    assert validator.check_node_inputs_has('TupleGetItem-7', ['Split-4', 1])
    assert validator.check_node_inputs_has('MakeTuple-2', ['TupleGetItem-6', 'TupleGetItem-7'])
    assert validator.check_node_inputs_has('Concat-1', ['MakeTuple-2', 2])
    assert validator.check_node_inputs_has('Reshape-0', ['Concat-1', '(2, 96, 256)'])
    assert validator.check_node_inputs_has('Split-5', ['Reshape-0', 0, 2])
    assert validator.check_node_inputs_has('TupleGetItem-8', ['Split-5', 0])
    assert validator.check_node_inputs_has('Split-6', ['TupleGetItem-8', 1, 2])
    assert validator.check_node_inputs_has('TupleGetItem-9', ['Split-6', 0])
    assert validator.check_node_inputs_has('ReLU-0', ['TupleGetItem-9'])


def test_pangu_multi_batch_qkv_reshape_scene_without_constant_folding():
    """
    Feature: Test tensor redistribution in dynamic shape.
    Description: Test PanGu multi-batch qkv reshape.
    Expectation: Compile success and assertion passed.
    """

    class PanguReshapeNet(nn.Cell):
        def __init__(self):
            super(PanguReshapeNet, self).__init__()
            self.add = P.Add().shard(((1, 8), (8,)))
            self.weight = Parameter(np.full((15360,), 0.5, dtype=np.float32), name="weight")
            self.reshape = P.Reshape()
            self.shape = P.Shape()
            self.transpose = P.Transpose().shard(((1, 1, 8, 1),))
            self.relu = P.ReLU().shard(((1, 8, 1, 1),))

        def construct(self, x):
            x = self.add(x, self.weight)  # (bs*seq, 15360), (8, -1, 120, 128)
            x = self.reshape(x, (self.shape(x)[0], -1, 120, 128))  # no constant folding
            x = self.transpose(x, (0, 2, 1, 3))
            x = self.relu(x)
            return x

    dump_ir_path = "./test_pangu_multi_batch_qkv_reshape_scene_without_constant_folding"
    context.set_context(save_graphs=True, save_graphs_path=dump_ir_path)
    dataset_shard = (1, 1)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8,
                                      dataset_strategy=(dataset_shard,))
    model = PanguReshapeNet()
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    x = Tensor(shape=[None, 15360], dtype=mstype.float32)
    model.set_inputs(x)
    phase = compile_net(model, x)
    validator = ParallelValidator(model, phase)
    assert validator.check_node_inputs_has('MakeTuple-0', ['inputs0'])
    assert validator.check_node_inputs_has('TupleGetItem-0', ['MakeTuple-0', 0])
    assert validator.check_node_inputs_has('Split-0', ['TupleGetItem-0', 1, 8])
    assert validator.check_node_inputs_has('TupleGetItem-1', ['Split-0', 0])
    assert validator.check_node_inputs_has('Add-0', ['TupleGetItem-1', 'Load-0'])
    assert validator.check_node_inputs_has('Shape-0', ['Add-0'])
    assert validator.check_node_inputs_has('TupleGetItem-2', ['Shape-0', 0])
    assert validator.check_node_inputs_has('MakeTuple-1', ['TupleGetItem-2', -1, 15, 128])
    assert validator.check_node_inputs_has('Reshape-0', ['Add-0', 'MakeTuple-1'])
    assert validator.check_node_inputs_has('Transpose-0', ['Reshape-0', '(0, 2, 1, 3)'])
    assert validator.check_node_inputs_has('ReLU-0', ['Transpose-0'])


def test_pangu_multi_batch_qkv_reshape_scene_with_constant_folding():
    """
    Feature: Test tensor redistribution in dynamic shape.
    Description: Test PanGu multi-batch qkv reshape.
    Expectation: Compile success and assertion passed.
    """

    class PanguReshapeNet(nn.Cell):
        def __init__(self):
            super(PanguReshapeNet, self).__init__()
            self.add = P.Add().shard(((1, 8), (8,)))
            self.weight = Parameter(np.full((15360,), 0.5, dtype=np.float32), name="weight")
            self.reshape = P.Reshape()
            self.shape = P.Shape()
            self.transpose = P.Transpose().shard(((1, 1, 8, 1),))
            self.relu = P.ReLU().shard(((1, 8, 1, 1),))

        def construct(self, x):
            x = self.add(x, self.weight)  # (bs*seq, 15360), (8, -1, 120, 128)
            x = self.reshape(x, (8, -1, 120, 128))  # constant folding
            x = self.transpose(x, (0, 2, 1, 3))
            x = self.relu(x)
            return x

    dump_ir_path = "./test_pangu_multi_batch_qkv_reshape_scene_with_constant_folding"
    context.set_context(save_graphs=True, save_graphs_path=dump_ir_path)
    dataset_shard = (1, 1)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8,
                                      dataset_strategy=(dataset_shard,))
    model = PanguReshapeNet()
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    x = Tensor(shape=(None, 15360), dtype=mstype.float32)
    model.set_inputs(x)
    phase = compile_net(model, x)
    validator = ParallelValidator(model, phase)
    assert validator.check_node_inputs_has('MakeTuple-0', ['inputs0'])
    assert validator.check_node_inputs_has('TupleGetItem-0', ['MakeTuple-0', 0])
    assert validator.check_node_inputs_has('Split-0', ['TupleGetItem-0', 1, 8])
    assert validator.check_node_inputs_has('TupleGetItem-1', ['Split-0', 0])
    assert validator.check_node_inputs_has('Add-0', ['TupleGetItem-1', 'Load-0'])
    assert validator.check_node_inputs_has('Reshape-0', ['Add-0', '(8, -1, 15, 128)'])
    assert validator.check_node_inputs_has('Transpose-0', ['Reshape-0', '(0, 2, 1, 3)'])
    assert validator.check_node_inputs_has('ReLU-0', ['Transpose-0'])

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
from mindspore import context, Tensor
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

    context.set_context(save_graphs=True, save_graphs_path="./no_need_to_accomplish")
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
    assert validator.check_node_inputs_has('Shape-0', ['TupleGetItem-0'])
    assert validator.check_node_inputs_has('tuple_getitem_for_value_3-0', ['Shape-0', 1])
    assert validator.check_node_inputs_has('MakeTuple-1', [2, 'tuple_getitem_for_value_3-0', 2, 2])
    assert validator.check_node_inputs_has('MakeTuple-2', [2, 'tuple_getitem_for_value_3-0', 1, 2])


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

    context.set_context(save_graphs=True, save_graphs_path="./no_need_to_accomplish_static")
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


def test_shape_used_by_one():
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

    context.set_context(save_graphs=True, save_graphs_path="./shape_used_by_one")
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
    assert validator.check_node_inputs_has('tuple_getitem_for_value_6-0', ['Shape-0', 1])
    assert validator.check_node_inputs_has('ScalarDiv-0', ['tuple_getitem_for_value_6-0', 2])
    assert validator.check_node_inputs_has('ScalarCast-0', ['ScalarDiv-0', 35])
    assert validator.check_node_inputs_has('MakeTuple-1', [4, 'ScalarCast-0', 2, 2])
    assert validator.check_node_inputs_has('StridedSlice-0', ['TupleGetItem-0', 'MakeTuple-1'])
    assert validator.check_node_inputs_has('tuple_getitem_for_value_6-1', ['Shape-1', 1])
    assert validator.check_node_inputs_has('MakeTuple-2', [4, 'tuple_getitem_for_value_6-1', 2, 1])
    assert validator.check_node_inputs_has('StridedSlice-1', ['Add-0', 'MakeTuple-2'])
    assert validator.check_node_inputs_has('MakeTuple-3', [4, 'tuple_getitem_for_value_6-1', 1, 1])
    assert validator.check_node_inputs_has('StridedSlice-2', ['StridedSlice-1', 'MakeTuple-3'])
    assert validator.check_node_inputs_has('ScalarMul-0', ['tuple_getitem_for_value_6-1', 2])
    assert validator.check_node_inputs_has('ScalarCast-1', ['ScalarMul-0', 35])
    assert validator.check_node_inputs_has('MakeTuple-5', [2, 'ScalarCast-1', 1, 1])
    assert validator.check_node_inputs_has('StridedSlice-3', ['Concat-0', 'MakeTuple-5'])


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

    context.set_context(save_graphs=True, save_graphs_path="./shape_used_by_one_static")
    dataset_shard = (1, 1, 1, 1)
    successor_shard = (2, 2, 2)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8,
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
            # 重排布插入的第一个reshape shape不对
            out = F.reshape(out, (shape[0] * shape[1], shape[2], shape[3]))  # 4,6,2,2
            out = self.relu(out)
            out = F.reshape(out, (shape[0] * shape[1], shape[2] * shape[3]))
            out = self.matmul(out, self.w)
            return out

    context.set_context(save_graphs=True, save_graphs_path="./shape_used_by_two")
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
    assert validator.check_node_inputs_has('MakeTuple-1', [1, 'tuple_getitem_for_value_6-0', 2, 2])
    assert validator.check_node_inputs_has('ScalarDiv-0', ['tuple_getitem_for_value_6-0', 2])
    assert validator.check_node_inputs_has('ScalarCast-0', ['ScalarDiv-0', 35])
    assert validator.check_node_inputs_has('MakeTuple-2', [1, 'ScalarCast-0', 2, 2])
    assert validator.check_node_inputs_has('MakeTuple-3', [1, 1, 'tuple_getitem_for_value_6-1', 2, 2])
    assert validator.check_node_inputs_has('AllGather-0', ['Reshape-0'])
    assert validator.check_node_inputs_has('ScalarMul-0', ['tuple_getitem_for_value_6-1', 2])
    assert validator.check_node_inputs_has('ScalarCast-1', ['ScalarMul-0', 35])
    assert validator.check_node_inputs_has('MakeTuple-5', [1, 1, 'ScalarCast-1', 2, 1])
    assert validator.check_node_inputs_has('StridedSlice-2', ['Concat-0', 'MakeTuple-5'])
    assert validator.check_node_inputs_has('AllGather-1', ['StridedSlice-2'])
    assert validator.check_node_inputs_has('ScalarMul-1', ['tuple_getitem_for_value_6-1', 2])
    assert validator.check_node_inputs_has('ScalarCast-2', ['ScalarMul-1', 35])
    assert validator.check_node_inputs_has('MakeTuple-7', [1, 2, 'ScalarCast-2', 1, 1])
    assert validator.check_node_inputs_has('StridedSlice-3', ['Concat-1', 'MakeTuple-7'])
    assert validator.check_node_inputs_fuzzy_match('Reshape-1', ['StridedSlice-3', '(-1, 1, 1)'])
    assert validator.check_node_inputs_fuzzy_match('Reshape-2', ['Concat-2', '(-1, 2)'])


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

    context.set_context(save_graphs=True, save_graphs_path="./shape_used_by_two_static")
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

    context.set_context(save_graphs=True, save_graphs_path="./two_dynamic_dims_used_by_two")
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
    _ = compile_net(model, input_ids)


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

    context.set_context(save_graphs=True, save_graphs_path="./two_dynamic_dims_used_by_two_static")
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
    assert validator.check_node_inputs_has('tuple_getitem_for_value_6-0', ['Shape-0', 1])
    assert validator.check_node_inputs_has('ScalarDiv-0', ['tuple_getitem_for_value_6-0', 2])
    assert validator.check_node_inputs_has('ScalarCast-0', ['ScalarDiv-0', 35])
    assert validator.check_node_inputs_has('MakeTuple-1', [4, 'ScalarCast-0'])
    assert validator.check_node_inputs_has('StridedSlice-0', ['TupleGetItem-0', 'MakeTuple-1'])
    assert validator.check_node_inputs_has('NotEqual-0', ['StridedSlice-0'])
    assert validator.check_node_inputs_has('Cast-0', ['NotEqual-0', 42])
    assert validator.check_node_inputs_has('AllGather-0', ['Cast-0'])
    assert validator.check_node_inputs_fuzzy_match('Reshape-0', ['Concat-0', '(-1)'])
    assert validator.check_node_inputs_has('Split-0', ['AllGather-0', 0, 2])
    assert validator.check_node_inputs_has('TupleGetItem-1', ['Split-0', 0])
    assert validator.check_node_inputs_has('TupleGetItem-2', ['Split-0', 1])
    assert validator.check_node_inputs_has('MakeTuple-2', ['TupleGetItem-1', 'TupleGetItem-2'])
    assert validator.check_node_inputs_has('Concat-0', ['MakeTuple-2', 1])


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
    assert validator.check_node_inputs_has('Shape-0', ['TupleGetItem-0'])
    assert validator.check_node_inputs_has('tuple_getitem_for_value_3-0', ['Shape-0', 1])
    assert validator.check_node_inputs_has('MakeTuple-1', [1, 'tuple_getitem_for_value_3-0'])
    assert validator.check_node_inputs_has('StridedSlice-0', ['TupleGetItem-0', 'MakeTuple-1'])
    assert validator.check_node_inputs_has('NotEqual-0', ['StridedSlice-0'])
    assert validator.check_node_inputs_has('Cast-0', ['NotEqual-0', 42])
    assert validator.check_node_inputs_fuzzy_match('Reshape-0', ['Cast-0', '(-1)'])


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
    _ = compile_net(model, input_ids)

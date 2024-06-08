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
from mindspore.nn.wrap.cell_wrapper import _VirtualDatasetCell
from mindspore import context, Tensor, Parameter
from mindspore.context import ParallelMode
from parallel.utils.utils import compile_net, ParallelValidator


def setup_function():
    context.reset_auto_parallel_context()
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


def test_parallel_dynamic_shape_with_features_007():
    """
    Feature: Test tensor redistribution in dynamic shape.
    Description: Corresponding test case is test_parallel_dynamic_shape_with_features_007.
    Expectation: Compile success and assertion passed.
    """

    class MatMulNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.matmul1 = P.MatMul()
            self.matmul2 = P.MatMul()
            self.matmul1_weight = Parameter(np.full((64, 32), 0.5, dtype=np.float32), name="weight1")
            self.matmul2_weight = Parameter(np.full((32, 32), 0.8, dtype=np.float32), name="weight2")
            self.matmul1.shard(((4, 1), (1, 2)))
            self.matmul2.shard(((2, 2), (2, 2)))
            self.relu = nn.ReLU()

        def construct(self, inputs):
            x = self.matmul1(inputs, self.matmul1_weight)
            x = self.matmul2(x, self.matmul2_weight)
            x = self.relu(x)
            return x

    dump_ir_path = "./test_parallel_dynamic_shape_with_features_007"
    context.set_context(save_graphs=True, save_graphs_path=dump_ir_path)
    dataset_shard = (1, 1)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8, full_batch=True,
                                      dataset_strategy=(dataset_shard,))
    model = MatMulNet()
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    s0 = Symbol(divisor=8)
    x = Tensor(shape=[s0, 64], dtype=mstype.float32)
    model.set_inputs(x)
    phase = compile_net(model, x)
    _ = ParallelValidator(model, phase)


def test_parallel_dynamic_shape_with_features_010():
    """
    Feature: Test tensor redistribution in dynamic shape.
    Description: Corresponding test case is test_parallel_dynamic_shape_with_features_010.
    Expectation: Compile success and assertion passed.
    """

    class ReShapeNet(nn.Cell):
        def __init__(self):
            super(ReShapeNet, self).__init__()
            self.weight = Parameter(np.full((1, 1, 1, 1), 0.5, dtype=np.float16), name="weight")
            self.add = P.Add().shard(((2, 2, 2, 1), (1, 1, 1, 1)))
            self.relu = P.ReLU().shard(((4, 1),))
            self.shape = P.Shape()
            self.reshape = P.Reshape()

        def construct(self, inputs):
            x = self.add(inputs, self.weight)  # shard: (2, 2, 2, 1)
            x_shape = self.shape(x)
            # in this scene, tensor redistribution should get reshape info
            x = self.reshape(x, (x_shape[0] * x_shape[1], x_shape[2] * x_shape[3]))
            x = self.relu(x)  # shard: (4, 1)
            return x

    dump_ir_path = "./test_parallel_dynamic_shape_with_features_010"
    context.set_context(save_graphs=True, save_graphs_path=dump_ir_path)
    dataset_shard = (1, 1, 1, 1)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8, full_batch=True,
                                      dataset_strategy=(dataset_shard,))
    model = ReShapeNet()
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    # 4,32,16,16
    s0 = Symbol(divisor=8)
    s1 = Symbol(divisor=4)
    x = Tensor(shape=[s0, 32, s1, 16], dtype=mstype.float16)
    model.set_inputs(x)
    phase = compile_net(model, x)
    _ = ParallelValidator(model, phase)


def test_parallel_reshape_has_multi_dynamic_axis():
    """
    Feature: Test tensor redistribution in dynamic shape.
    Description: Reshape has more than one dynamic axis.
    Expectation: Compile success and assertion passed.
    """

    class ReshapeNet(nn.Cell):
        def __init__(self):
            super(ReshapeNet, self).__init__()
            self.relu0 = P.ReLU().shard(((2, 4, 1, 1),))
            self.relu1 = P.ReLU().shard(((8, 1),))
            self.shape = P.Shape()
            self.reshape = P.Reshape()

        def construct(self, inputs):
            x = self.relu0(inputs)  # shard: (2, 4, 1, 1)
            x_shape = self.shape(x)
            x = self.reshape(x, (x_shape[0] * x_shape[1], x_shape[2] * x_shape[3]))
            x = self.relu1(x)  # shard: (8, 1)
            return x

    dump_ir_path = "./test_parallel_reshape_has_multi_dynamic_axis"
    context.set_context(save_graphs=True, save_graphs_path=dump_ir_path)
    dataset_shard = (1, 1, 1, 1)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8, full_batch=True,
                                      dataset_strategy=(dataset_shard,))
    model = ReshapeNet()
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    # 4,32,16,16
    s0 = Symbol(divisor=8)
    s1 = Symbol(divisor=4)
    x = Tensor(shape=[s0, 32, s1, 16], dtype=mstype.float16)
    model.set_inputs(x)
    phase = compile_net(model, x)
    _ = ParallelValidator(model, phase)


def test_parallel_static_reshape_has_multi_user():
    """
    Feature: Test tensor redistribution in dynamic shape.
    Description: Reshape has more than one dynamic axis.
    Expectation: Compile success and assertion passed.
    """

    class ReshapeNet(nn.Cell):
        def __init__(self):
            super(ReshapeNet, self).__init__()
            self.relu0 = P.ReLU().shard(((2, 4),))
            self.relu1 = P.ReLU().shard(((4, 2),))
            self.shape = P.Shape()
            self.reshape = P.Reshape()

        def construct(self, inputs):
            x_shape = self.shape(inputs)
            x = self.reshape(inputs, (x_shape[0] * x_shape[1], x_shape[2] * x_shape[3]))
            y1 = self.relu0(x)  # shard: (2, 4)
            y2 = self.relu1(x)  # shard: (4, 2)
            return y1 + y2

    dump_ir_path = "./test_parallel_static_reshape_has_multi_user"
    context.set_context(save_graphs=True, save_graphs_path=dump_ir_path)
    dataset_shard = (1, 1, 1, 1)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8, full_batch=True,
                                      dataset_strategy=(dataset_shard,))
    model = ReshapeNet()
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    # 4,32,16,16
    # s0 = Symbol(divisor=8)
    # s1 = Symbol(divisor=4)
    x = Tensor(np.random.rand(3, 32, 5, 16), dtype=mstype.float16)
    model.set_inputs(x)
    phase = compile_net(model, x)
    _ = ParallelValidator(model, phase)

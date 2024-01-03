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
"""PanguAlpha model"""
import pytest
import numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.wrap.cell_wrapper import _VirtualDatasetCell
from mindspore import context, Tensor
from mindspore.context import ParallelMode
from parallel.utils.utils import ParallelValidator, compile_net

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


def setup_function():
    context.reset_auto_parallel_context()


class TestRedistribution(nn.Cell):
    def __init__(self, from_shard=None, to_shard=None):
        super(TestRedistribution, self).__init__()
        self.start_node = P.ReLU().shard((from_shard,))  # (1, 2, 1, 1)->(1, 2, 1, 1, 4)
        self.dist_node = P.ReLU().shard((to_shard,))

    def construct(self, x):
        out = self.start_node(x)
        out = self.dist_node(out)
        return out


class TestTensorShape(nn.Cell):
    def __init__(self):
        super(TestTensorShape, self).__init__()
        self.relu = P.ReLU()

    def construct(self, x):
        out = self.relu(x)
        shape = F.shape(out)
        out0 = F.reshape(out, (shape[0] * shape[1], shape[2] * shape[3]))
        return out0


def test_shape_ir_static():
    """
    Feature: Tensor distribution with dynamic shape.
    Description: Test generate ir with static shape.
    Expectation: Compile success.
    """
    context.set_context(save_graphs=True, save_graphs_path="./shape_ir_static")
    model = TestTensorShape().to_float(mstype.float16)
    input_ids = Tensor(np.random.rand(4, 768, 2, 2), dtype=mstype.float16)
    model.set_inputs(input_ids)
    model.compile(input_ids)


def test_shape_ir_dyn():
    """
    Feature: Tensor distribution with dynamic shape.
    Description: Test generate ir with dynamic shape.
    Expectation: Compile success.
    """
    context.set_context(save_graphs=True, save_graphs_path="./shape_ir_dyn")
    model = TestTensorShape().to_float(mstype.float16)
    input_ids = Tensor(shape=[4, None, 2, 2], dtype=mstype.float16)
    model.set_inputs(input_ids)
    model.compile(input_ids)


def test_all2all_static():
    """
    Feature: Tensor distribution with dynamic shape.
    Description: Test all2all ir compiling with static shape.
    Expectation: Compile success.
    """
    # pre_node: @:CNode_4{[0]: ValueNode<PrimitivePy> ReLU, [1]: CNode_4}
    # next_node: @:CNode_4{[0]: ValueNode<PrimitivePy> ReLU, [1]: CNode_4}
    # device arrangement = [ 2 4 ]
    # tensor map = [ -1 1 -1 -1 ]
    # tensor shape = [ 1 1536 2 2 ]
    # device arrangement origin = [ 1 2 1 1 4 ]
    # tensor map origin = [ 4 3 2 1 ]
    # tensor shape origin = [ 1 1536 2 2 ]
    # ------------>>>>>>>>>
    # device arrangement = [ 2 4 ]
    # tensor map = [ -1 -1 1 -1 ]
    # tensor shape = [ 1 1536 2 2 ]
    # device arrangement origin = [ 1 1 2 1 4 ]
    # tensor map origin = [ 4 3 2 1 ]
    # tensor shape origin = [ 1 1536 2 2 ]
    context.reset_auto_parallel_context()
    from_shard = (1, 2, 1, 1)
    to_shard = (1, 1, 2, 1)

    context.set_context(save_graphs=True, save_graphs_path="./alltoall_ir_static")
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, global_rank=0, device_num=8,
                                      dataset_strategy=(from_shard,))
    model = TestRedistribution(from_shard, to_shard).to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    input_ids = Tensor(np.random.rand(3, 768, 2, 2), dtype=mstype.float16)
    phase = compile_net(model, input_ids)
    validator = ParallelValidator(model, phase)
    context.reset_auto_parallel_context()
    assert validator.check_node_inputs_has('ReLU-1', ['StridedSlice-0'])


def test_all2all_dyn():
    """
    Feature: Tensor distribution with dynamic shape.
    Description: Test all2all ir compiling with dynamic shape.
    Expectation: Compile success.
    """
    # pre_node: @:CNode_4{[0]: ValueNode<PrimitivePy> ReLU, [1]: CNode_4}
    # next_node: @:CNode_4{[0]: ValueNode<PrimitivePy> ReLU, [1]: CNode_4}
    # device arrangement = [ 2 4 ]
    # tensor map = [ -1 1 -1 -1 ]
    # tensor shape = [ -1 1536 2 2 ]
    # device arrangement origin = [ 1 2 1 1 4 ]
    # tensor map origin = [ 4 3 2 1 ]
    # tensor shape origin = [ -1 1536 2 2 ]
    # ------------>>>>>>>>>
    # device arrangement = [ 2 4 ]
    # tensor map = [ -1 -1 1 -1 ]
    # tensor shape = [ -1 1536 2 2 ]
    # device arrangement origin = [ 1 1 2 1 4 ]
    # tensor map origin = [ 4 3 2 1 ]
    # tensor shape origin = [ -1 1536 2 2 ]
    # all2all: Reshape->AllGather->StridedSlice->Reshape
    context.reset_auto_parallel_context()
    from_shard = (1, 2, 1, 1)
    to_shard = (1, 1, 2, 1)

    context.set_context(save_graphs=True, save_graphs_path="./alltoall_ir_dyn")
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, global_rank=0, device_num=8,
                                      dataset_strategy=(from_shard,))
    # _VirtualDataset，一定要做数据数据并行？(8,1,1,1)
    model = TestRedistribution(from_shard, to_shard).to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    input_ids = Tensor(shape=[None, 768, 2, 2], dtype=mstype.float16)
    phase = compile_net(model, input_ids)
    validator = ParallelValidator(model, phase)
    context.reset_auto_parallel_context()
    assert validator.check_node_inputs_has('ReLU-1', ['StridedSlice-0'])


def test_allconcat_static():
    """
    Feature: Tensor distribution with dynamic shape.
    Description: Test allconcat ir compiling with static shape.
    Expectation: Compile success.
    """
    # allconcat: Reshape->AllGather->Reshape
    context.reset_auto_parallel_context()
    from_shard = (1, 2, 1, 1)
    to_shard = (1, 1, 1, 1)

    context.set_context(save_graphs=True, save_graphs_path="./allconcat_ir_static")
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, global_rank=0, device_num=8,
                                      dataset_strategy=(from_shard,))
    # _VirtualDataset，一定要做数据数据并行？(8,1,1,1)
    model = TestRedistribution(from_shard, to_shard).to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    input_ids = Tensor(np.random.rand(5, 768, 2, 2), dtype=mstype.float16)
    phase = compile_net(model, input_ids)
    validator = ParallelValidator(model, phase)
    context.reset_auto_parallel_context()
    assert validator.check_node_inputs_has('ReLU-1', ['Concat-0'])


def test_allconcat_dyn():
    """
    Feature: Tensor distribution with dynamic shape.
    Description: Test allconcat ir compiling with dynamic shape.
    Expectation: Compile success.
    """
    # allconcat: Reshape->AllGather->Reshape
    # AllGather->PrimFunc_ReLU
    # AllGather->PrimFunc_Split->TupleGetItem->MakeTuple->PrimFunc_Concat->PrimFunc_ReLU
    context.reset_auto_parallel_context()
    from_shard = (1, 2, 1, 1)
    to_shard = (1, 1, 1, 1)

    context.set_context(save_graphs=True, save_graphs_path="./allconcat_ir_dyn")
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, global_rank=0, device_num=8,
                                      dataset_strategy=(from_shard,))
    # _VirtualDataset，一定要做数据数据并行？(8,1,1,1)
    model = TestRedistribution(from_shard, to_shard).to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    input_ids = Tensor(shape=[None, 768, 2, 2], dtype=mstype.float16)
    phase = compile_net(model, input_ids)
    validator = ParallelValidator(model, phase)
    context.reset_auto_parallel_context()
    assert validator.check_node_inputs_has('ReLU-1', ['Concat-0'])


def test_allsplit_static():
    """
    Feature: Tensor distribution with dynamic shape.
    Description: Test allsplit ir compiling with static shape.
    Expectation: Compile success.
    """
    # allsplit: Reshape->StridedSlice->Reshape
    context.reset_auto_parallel_context()
    from_shard = (1, 1, 1, 1)
    to_shard = (1, 2, 1, 1)

    context.set_context(save_graphs=True, save_graphs_path="./allsplit_ir_static")
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, global_rank=0, device_num=8,
                                      dataset_strategy=(from_shard,))
    # _VirtualDataset，一定要做数据数据并行？(8,1,1,1)
    model = TestRedistribution(from_shard, to_shard).to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    input_ids = Tensor(np.random.rand(1, 768, 2, 2), dtype=mstype.float16)
    phase = compile_net(model, input_ids)
    validator = ParallelValidator(model, phase)
    context.reset_auto_parallel_context()
    assert validator.check_node_inputs_has('ReLU-1', ['Reshape-1'])
    assert validator.check_node_inputs_has('Reshape-1', ['StridedSlice-0'])


def test_allsplit_dyn():
    """
    Feature: Tensor distribution with dynamic shape.
    Description: Test allsplit ir compiling with dynamic shape.
    Expectation: Compile success.
    """
    # allsplit: Reshape->StridedSlice->Reshape
    # StridedSlice的End位置有问题
    context.reset_auto_parallel_context()
    from_shard = (1, 1, 1, 1)
    to_shard = (1, 2, 1, 1)

    context.set_context(save_graphs=True, save_graphs_path="./allsplit_ir_dyn")
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, global_rank=0, device_num=8,
                                      dataset_strategy=(from_shard,))
    # _VirtualDataset，一定要做数据数据并行？(8,1,1,1)
    model = TestRedistribution(from_shard, to_shard).to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    input_ids = Tensor(shape=[None, 768, 2, 2], dtype=mstype.float16)
    phase = compile_net(model, input_ids)
    validator = ParallelValidator(model, phase)
    context.reset_auto_parallel_context()
    assert validator.check_node_inputs_has('ReLU-1', ['StridedSlice-0'])


def test_allsplit_parallel_on_dynamic_dim():
    """
    Feature: Tensor distribution with dynamic shape.
    Description: Test allsplit ir compiling with dynamic shape.
    Expectation: Compile success.
    """
    # allsplit: Reshape->StridedSlice->Reshape
    context.reset_auto_parallel_context()
    from_shard = (1, 1, 1, 1)
    to_shard = (2, 1, 1, 1)

    context.set_context(save_graphs=True, save_graphs_path="./allsplit_parallel_on_dyn_dim_ir")
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8,
                                      dataset_strategy=(from_shard,))
    model = TestRedistribution(from_shard, to_shard).to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    input_ids = Tensor(shape=[None, 768, 2, 2], dtype=mstype.float16)
    phase = compile_net(model, input_ids)
    validator = ParallelValidator(model, phase)
    context.reset_auto_parallel_context()
    assert validator.check_node_inputs_has('ReLU-1', ['StridedSlice-0'])


def test_allsplit_parallel_on_static_batch_dim():
    """
    Feature: Tensor distribution with dynamic shape.
    Description: Test allsplit ir compiling with dynamic shape.
    Expectation: Compile success.
    """
    # allsplit: Reshape->StridedSlice->Reshape
    context.reset_auto_parallel_context()
    from_shard = (1, 1, 1, 1)
    to_shard = (2, 1, 1, 1)

    context.set_context(save_graphs=True, save_graphs_path="./allsplit_parallel_on_static_batch_dim_ir")
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8,
                                      dataset_strategy=(from_shard,))
    model = TestRedistribution(from_shard, to_shard).to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    input_ids = Tensor(np.random.rand(8, 768, 2, 2), dtype=mstype.float16)
    phase = compile_net(model, input_ids)
    validator = ParallelValidator(model, phase)
    context.reset_auto_parallel_context()
    assert validator.check_node_inputs_has('ReLU-1', ['StridedSlice-0'])


def test_allconcat_parallel_on_dynamic_seq_dim():
    """
    Feature: Tensor distribution with dynamic shape.
    Description: Test allconcat ir compiling with dynamic shape.
    Expectation: Compile success.
    """
    context.reset_auto_parallel_context()
    from_shard = (1, 4, 1, 1)
    to_shard = (1, 1, 1, 1)

    context.set_context(save_graphs=True, save_graphs_path="./allconcat_parallel_on_dyn_seq_dim_ir")
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8,
                                      dataset_strategy=(from_shard,))
    model = TestRedistribution(from_shard, to_shard).to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    input_ids = Tensor(shape=[8, None, 2, 2], dtype=mstype.float16)
    phase = compile_net(model, input_ids)
    validator = ParallelValidator(model, phase)
    context.reset_auto_parallel_context()
    assert validator.check_node_inputs_has('ReLU-1', ['Concat-0'])


def test_allconcat_parallel_on_static_seq_dim():
    """
    Feature: Tensor distribution with dynamic shape.
    Description: Test allconcat ir compiling with dynamic shape.
    Expectation: Compile success.
    """
    # allsplit: Reshape->StridedSlice->Reshape
    context.reset_auto_parallel_context()
    from_shard = (1, 4, 1, 1)
    to_shard = (1, 1, 1, 1)

    context.set_context(save_graphs=True, save_graphs_path="./allconcat_parallel_on_static_seq_dim_ir")
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8,
                                      dataset_strategy=(from_shard,))
    model = TestRedistribution(from_shard, to_shard).to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    input_ids = Tensor(np.random.rand(8, 768, 2, 2), dtype=mstype.float16)
    phase = compile_net(model, input_ids)
    validator = ParallelValidator(model, phase)
    context.reset_auto_parallel_context()
    assert validator.check_node_inputs_has('ReLU-1', ['Concat-0'])


# @pytest.mark.skip(reason="offline this testcase for tensor redistribution temporarily.")
def test_all2all_parallel_on_dynamic_seq_dim():
    """
    Feature: Tensor distribution with dynamic shape.
    Description: Test all2all ir compiling with dynamic shape.
    Expectation: Compile success.
    """
    context.reset_auto_parallel_context()
    from_shard = (1, 2, 1, 1)
    to_shard = (1, 1, 2, 1)

    context.set_context(save_graphs=True, save_graphs_path="./all2all_parallel_on_dyn_seq_dim_ir")
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8,
                                      dataset_strategy=(from_shard,))
    model = TestRedistribution(from_shard, to_shard).to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    input_ids = Tensor(shape=[8, None, 2, 2], dtype=mstype.float16)
    phase = compile_net(model, input_ids)
    validator = ParallelValidator(model, phase)
    context.reset_auto_parallel_context()
    assert validator.check_node_inputs_has('ReLU-1', ['StridedSlice-0'])


def test_all2all_parallel_on_static_seq_dim():
    """
    Feature: Tensor distribution with dynamic shape.
    Description: Test all2all ir compiling with dynamic shape.
    Expectation: Compile success.
    """
    context.reset_auto_parallel_context()
    from_shard = (1, 2, 1, 1)
    to_shard = (1, 1, 2, 1)

    context.set_context(save_graphs=True, save_graphs_path="./all2all_parallel_on_static_seq_dim_ir")
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8,
                                      dataset_strategy=(from_shard,))
    model = TestRedistribution(from_shard, to_shard).to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    input_ids = Tensor(np.random.rand(8, 768, 2, 2), dtype=mstype.float16)
    phase = compile_net(model, input_ids)
    validator = ParallelValidator(model, phase)
    context.reset_auto_parallel_context()
    assert validator.check_node_inputs_has('ReLU-1', ['StridedSlice-0'])


@pytest.mark.skip(reason="offline this testcase for tensor redistribution temporarily.")
def test_allsplit_parallel_on_multi_dynamic_dims():
    """
    Feature: Tensor distribution with dynamic shape.
    Description: Test allsplit on multi dim ir compiling with dynamic shape.
    Expectation: Compile success.
    """
    # TODO: Need to insert TensorShape+TupleGetItem+MakeTuple to construct graph.
    context.reset_auto_parallel_context()
    from_shard = (1, 1, 1, 1)
    to_shard = (1, 2, 2, 2)

    context.set_context(save_graphs=True, save_graphs_path="./test_allsplit_parallel_on_multi_dynamic_dim_ir")
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8,
                                      dataset_strategy=(from_shard,))
    model = TestRedistribution(from_shard, to_shard).to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    input_ids = Tensor(shape=[None, None, 8, None], dtype=mstype.float16)
    model.set_inputs(input_ids)
    model.compile(input_ids)
    context.reset_auto_parallel_context()


@pytest.mark.skip(reason="offline this testcase for tensor redistribution temporarily.")
def test_mixed_parallel_on_dynamic_shape_dim():
    """
    Feature: Tensor distribution with dynamic shape.
    Description: Test mix tensor redistribution ir compiling with dynamic shape.
    Expectation: Compile success.
    """
    context.reset_auto_parallel_context()
    dataset_shard = (1, 2, 1, 1)
    from_shard = (1, 2, 4, 1)
    to_shard = (1, 4, 1, 2)

    context.set_context(save_graphs=True, save_graphs_path="./alltoall_alltoall_ir_dyn")
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8, dataset_strategy=(dataset_shard,))
    model = TestRedistribution(from_shard, to_shard).to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    input_ids = Tensor(shape=[1, None, 4, 4], dtype=mstype.float16)
    model.set_inputs(input_ids)
    model.compile(input_ids)
    context.reset_auto_parallel_context()

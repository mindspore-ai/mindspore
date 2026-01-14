# Copyright 2025 Huawei Technologies Co., Ltd
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
# ==============================================================================
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import context, Tensor, ops
from mindspore.common import dtype as mstype
from mindspore.ops.auto_generate import TransposeView
from mindspore._c_expression import UMonad
from tests.st.backend.ms_backend.common.backend_graph import BackendGraph
from tests.mark_utils import arg_mark
import pytest

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.00)
def test_output_view_parameter():
    """
    Feature: Support view.
    Description: Support view.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.atleast_2d = ops.atleast_2d

        def construct(self, *args):
            return self.atleast_2d(args)

    input_x = Tensor(np.random.randn(4, 4, 4), mstype.float64)
    net = Net()
    context.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})
    res1 = net(*input_x)
    context.set_context(mode=ms.PYNATIVE_MODE)
    res2 = net(*input_x)
    assert np.allclose(res1[3].asnumpy(), res2[3].asnumpy())

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_inplace_ops_with_view_input():
    """
    Feature: Inplace operator doesn't accept a view input
    Description: Runtime throws an exception when an inplace operator uses a view input
    Expectation: No exception.
    """
    def func():
        tensor = Tensor(np.ones((3, 2), dtype=np.float32))
        view_tensor = TransposeView()(tensor, (1, 0))
        assign_add = ops.AssignAdd()
        assign_add(view_tensor, Tensor(np.ones((2, 3), dtype=np.float32)))

    with pytest.raises(RuntimeError) as err:
        func_jit = ms.jit(func, backend="ms_backend")
        func_jit()
    assert "is an inplace op and does not support view input." in str(err.value)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
def test_view_to_only_depend_shape():
    """
    Feature: view output to shape op.
    Description: not convert contiguous.
    Expectation: No exception.
    """
    context.set_context(jit_config={"jit_level": "O1"})
    root_graph = BackendGraph()
    a = BackendGraph()
    func_a = root_graph.add_valuenode(a)

    root_para_1 = root_graph.add_parameter(ms.float32, (-1, -1))
    root_para_2 = root_graph.add_parameter(ms.float32, (-1,))
    root_para_3 = root_graph.add_parameter(ms.float32, (-1, -1))
    U = root_graph.add_valuenode(UMonad())
    root_add = root_graph.add_cnode("Add", root_para_3, root_para_3)
    root_scalar_0 = root_graph.add_valuenode(0)
    select_ext = root_graph.add_cnode("SelectExtView", root_add, root_scalar_0, root_scalar_0, U)
    call = root_graph.add_cnode(func_a, root_para_1, root_para_2, select_ext)
    root_graph.add_return(call)

    para_1 = a.add_parameter(ms.float32, (-1, -1))
    para_2 = a.add_parameter(ms.float32, (-1,))
    para_3 = a.add_parameter(ms.float32, (-1,))

    shape_1 = a.add_cnode("Shape", para_3)
    get_item = a.add_cnode("TupleGetItem", shape_1, a.add_valuenode(0))
    make_tuple_1 = a.add_cnode("MakeTuple", a.add_valuenode(10), get_item)
    scalar_0 = a.add_valuenode(0)
    tuple_0_0 = a.add_valuenode((0, 0))
    tuple_1_1 = a.add_valuenode((1, 1))
    strided_slice = a.add_cnode("StridedSlice", para_1, tuple_0_0, make_tuple_1, tuple_1_1, scalar_0, scalar_0,
                                scalar_0, scalar_0, scalar_0)
    ones_like = a.add_cnode("OnesLike", strided_slice)
    shape_2 = a.add_cnode("Shape", para_1)
    strided_slice_grad = a.add_cnode("StridedSliceGrad", ones_like, shape_2, tuple_0_0, make_tuple_1, tuple_1_1)
    zeros_like = a.add_cnode("ZerosLike", para_2)
    make_tuple_2 = a.add_cnode("MakeTuple", strided_slice_grad, zeros_like)
    a.add_return(make_tuple_2)

    root_graph.infer()
    print(root_graph)
    root_graph.compile()
    p1 = np.random.rand(32, 16).astype(np.float32)
    p2 = np.random.rand(10).astype(np.float32)
    p3 = np.random.rand(32, 10).astype(np.float32)
    out = root_graph(Tensor(p1), Tensor(p2), Tensor(p3))
    print(out)

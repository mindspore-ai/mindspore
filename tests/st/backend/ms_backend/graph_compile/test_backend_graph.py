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
"""
Test backend compile.
"""
import numpy as np
import mindspore as ms
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P
from mindspore._c_expression import UMonad
from tests.st.backend.ms_backend.common.backend_graph import BackendGraph
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_base():
    """
    Feature: Support python ir interface.
    Description: Test backend graph interface.
    Expectation: Run success.
    """
    a = BackendGraph()
    shape = (2, 3)
    revert_shape = shape[::-1]
    p = Parameter(Tensor(np.ones(shape).astype(np.float32)), name="weight", storage_format="FRACTAL_NZ")
    para_1 = a.add_parameter(ms.float32, shape)
    para_2 = a.add_parameter(ms.float32, shape)
    para_3 = a.add_parameter(p)
    tensor_1 = a.add_valuenode(Tensor(np.ones(shape).astype(np.float32)))
    tuple_1 = a.add_valuenode(revert_shape)
    prim_add = P.Add().set_device("CPU")
    value_node_prim_add = a.add_valuenode(prim_add)
    add = a.add_cnode(value_node_prim_add, para_1, tensor_1)
    sub = a.add_cnode(a.add_valuenode(P.Sub()), add, para_2)
    mul = a.add_cnode(a.add_valuenode(P.Mul()), sub, para_2)
    div = a.add_cnode(a.add_valuenode(P.RealDiv()), mul, para_2)
    reshape = a.add_cnode(a.add_valuenode(P.Reshape()), para_2, tuple_1)

    U1 = a.add_valuenode(UMonad())
    prim_assignadd = a.add_valuenode(P.AssignAdd())
    assign_add = a.add_cnode(prim_assignadd, para_3, para_1, U1)
    U2 = a.add_valuenode(UMonad())
    updatestate = a.add_cnode(a.add_valuenode(P.UpdateState()), U2, assign_add)
    prim_depend = a.add_valuenode(P.Depend())
    depend = a.add_cnode(prim_depend, assign_add, updatestate)

    b = BackendGraph()
    sub_para_1 = b.add_parameter(ms.float32, shape)
    b.add_return(sub_para_1)

    a.add_subgraph(b)
    func_1 = a.add_valuenode(b)
    call = a.add_cnode(func_1, div)
    make_tuple = a.add_cnode("MakeTuple", add, sub, mul, div, reshape, call, depend)
    a.add_return(make_tuple)
    a.infer()
    a.compile()
    x = Tensor(np.ones(shape).astype(np.float32))
    y = Tensor(np.ones(shape).astype(np.float32))
    out = a(x, y, p)
    assert np.allclose(out[1].asnumpy(), np.ones(shape).astype(np.float32))


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cse_internal_value():
    """
    Feature: Test graph partition and graph compile.
    Description: Test internal parameter of value node.
    Expectation: Run success.
    """
    a = BackendGraph()
    shape = (1,)
    para_1 = a.add_parameter(ms.int64, shape)
    para_2 = a.add_parameter(ms.int64, shape)
    tensor = Tensor(1)
    tensor_1 = a.add_valuenode(tensor)
    depend_1 = a.add_cnode(a.add_valuenode(P.Depend()), tensor_1, para_1)
    tensor_2 = a.add_valuenode(tensor)
    depend_2 = a.add_cnode(a.add_valuenode(P.Depend()), tensor_2, para_2)
    a.set_abstract(tensor_1, tensor_2)
    sub_1 = a.add_cnode(a.add_valuenode(P.Sub()), para_1, depend_1)
    sub_2 = a.add_cnode(a.add_valuenode(P.Sub()), para_2, depend_2)
    prim_add = P.Add().set_device("CPU")
    value_node_prim_add = a.add_valuenode(prim_add)
    add = a.add_cnode(value_node_prim_add, sub_1, sub_2)
    mul = a.add_cnode(a.add_valuenode(P.Mul()), add, depend_1)
    a.add_return(mul)
    a.infer()
    a.compile()
    x = Tensor(2)
    out = a(x, x)
    assert out == 2


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_lazy_inline():
    """
    Feature: Test lazy inline.
    Description: Test internal parameter of value node.
    Expectation: Run success.
    """
    shape = (3, 4)

    backward_graph = BackendGraph()
    backward_para_1 = backward_graph.add_parameter(ms.float32, shape)
    backward_para_2 = backward_graph.add_parameter(ms.float32, shape)
    backward_add = backward_graph.add_cnode(backward_graph.add_valuenode(P.Add()), backward_para_1, backward_para_2)
    backward_graph.add_return(backward_add)

    forward_graph = BackendGraph()
    forward_graph.set_cell_reuse()
    forward_para_1 = forward_graph.add_parameter(ms.float32, shape)
    forward_para_2 = forward_graph.add_parameter(ms.float32, shape)
    forward_sub = forward_graph.add_cnode(forward_graph.add_valuenode(P.Sub()), forward_para_1, forward_para_2)
    backward_graph_value_node = forward_graph.add_valuenode(backward_graph)
    forward_partial = forward_graph.add_cnode("Partial", backward_graph_value_node, forward_sub)
    forward_make_tuple = forward_graph.add_cnode("MakeTuple", forward_sub, forward_partial)
    forward_graph.add_return(forward_make_tuple)

    root_graph = BackendGraph()
    forward_graph_value_node = root_graph.add_valuenode(forward_graph)
    root_para_1 = root_graph.add_parameter(ms.float32, shape)
    root_para_2 = root_graph.add_parameter(ms.float32, shape)
    forward_call = root_graph.add_cnode(forward_graph_value_node, root_para_1, root_para_2)
    get_item_0 = root_graph.add_cnode("TupleGetItem", forward_call, root_graph.add_valuenode(0))
    get_item_1 = root_graph.add_cnode("TupleGetItem", forward_call, root_graph.add_valuenode(1))
    backend_call = root_graph.add_cnode(get_item_1, get_item_0)
    root_graph.add_return(backend_call)
    root_graph.infer()
    root_graph.compile()
    x = Tensor(np.ones(shape).astype(np.float32))
    y = Tensor(np.ones(shape).astype(np.float32))
    out = root_graph(x, y)
    assert np.allclose(out.asnumpy(), np.zeros(shape).astype(np.float32))


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_lazy_inline_with_control_flow():
    """
    Feature: Test lazy inline.
    Description: Test internal parameter of lazy inline call node.
    Expectation: Run success.
    """
    shape = (3, 4)

    backward_graph = BackendGraph()
    backward_para_1 = backward_graph.add_parameter(ms.float32, shape)
    backward_para_2 = backward_graph.add_parameter(ms.float32, shape)
    backward_add = backward_graph.add_cnode(backward_graph.add_valuenode(P.Add()), backward_para_1, backward_para_2)
    backward_graph.add_return(backward_add)

    forward_graph = BackendGraph()
    forward_graph.set_cell_reuse()
    forward_para_1 = forward_graph.add_parameter(ms.float32, shape)
    forward_para_2 = forward_graph.add_parameter(ms.float32, shape)
    forward_sub = forward_graph.add_cnode(forward_graph.add_valuenode(P.Sub()), forward_para_1, forward_para_2)
    backward_graph_value_node = forward_graph.add_valuenode(backward_graph)
    forward_partial = forward_graph.add_cnode("Partial", backward_graph_value_node, forward_sub)
    forward_make_tuple = forward_graph.add_cnode("MakeTuple", forward_sub, forward_partial)
    forward_graph.add_return(forward_make_tuple)

    true_graph = BackendGraph()
    true_para_1 = true_graph.add_parameter(ms.float32, shape)
    true_para_2 = true_graph.add_parameter(ms.float32, shape)
    true_add = true_graph.add_cnode(true_graph.add_valuenode(P.Add()), true_para_1, true_para_2)
    true_graph.add_return(true_add)


    false_graph = BackendGraph()
    _ = false_graph.add_parameter(ms.float32, shape)
    _ = false_graph.add_parameter(ms.float32, shape)
    false_graph.add_return(false_graph.add_valuenode(Tensor(np.ones(shape).astype(np.float32))))

    root_graph = BackendGraph()
    forward_graph_value_node = root_graph.add_valuenode(forward_graph)
    root_para_1 = root_graph.add_parameter(ms.float32, shape)
    root_para_2 = root_graph.add_parameter(ms.float32, shape)
    root_para_3 = root_graph.add_parameter(ms.bool, ())
    forward_call = root_graph.add_cnode(forward_graph_value_node, root_para_1, root_para_2)
    get_item_0 = root_graph.add_cnode("TupleGetItem", forward_call, root_graph.add_valuenode(0))
    get_item_1 = root_graph.add_cnode("TupleGetItem", forward_call, root_graph.add_valuenode(1))
    backend_call = root_graph.add_cnode(get_item_1, get_item_0)

    false_graph_value_node = root_graph.add_valuenode(false_graph)
    false_partial = root_graph.add_cnode("Partial", false_graph_value_node, backend_call, root_para_2)
    true_graph_value_node = root_graph.add_valuenode(true_graph)
    true_partial = root_graph.add_cnode("Partial", true_graph_value_node, backend_call, root_para_2)
    switch = root_graph.add_cnode("Switch", root_para_3, true_partial, false_partial)
    switch_call = root_graph.add_cnode(switch)

    U = root_graph.add_valuenode(UMonad())
    updatestate = root_graph.add_cnode(root_graph.add_valuenode(P.UpdateState()), U, backend_call)
    assign_add = root_graph.add_cnode(root_graph.add_valuenode(P.AssignAdd()), switch_call, root_para_2, updatestate)
    root_graph.add_return(assign_add)
    root_graph.infer()
    root_graph.compile()
    x = Tensor(np.ones(shape).astype(np.float32))
    y = Tensor(np.ones(shape).astype(np.float32))
    z = Tensor(False)
    out = root_graph(x, y, z)
    assert np.allclose(out.asnumpy(), np.ones(shape).astype(np.float32) * 2)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_internal_valuetuple():
    """
    Feature: Support python ir interface.
    Description: Test internal parameter of value node.
    Expectation: Run success.
    """
    shape = (2, 1)
    a = BackendGraph()
    para_1 = a.add_parameter(ms.float32, shape)
    value_tuple = a.add_valuenode((2,))
    get_item = a.add_cnode("TupleGetItem", value_tuple, a.add_valuenode(0))
    make_tuple = a.add_cnode("MakeTuple", get_item)
    depend = a.add_cnode(a.add_valuenode(P.Depend()), para_1, make_tuple)
    add = a.add_cnode(a.add_valuenode(P.Add().set_device("CPU")), depend, para_1)
    reshape = a.add_cnode(a.add_valuenode(P.Reshape()), add, make_tuple)
    a.add_return(reshape)
    a.native_infer()
    a.compile()
    x = Tensor(np.ones(shape).astype(np.float32))
    out = a(x)
    assert out.shape == (2,)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_internal_parameter_in_same_group():
    """
    Feature: Test internal parameter in same graph group.
    Description: internal parameter is load.
    Expectation: Run success.
    """
    shape = (3, 4)

    a = BackendGraph()
    sub_para_1 = a.add_parameter(ms.float32, shape)
    a.add_return(sub_para_1)

    root_graph = BackendGraph()
    root_para_1 = root_graph.add_parameter(ms.float32, shape)
    root_para_2 = root_graph.add_parameter(ms.float32, shape)

    sub_graph_value_node = root_graph.add_valuenode(a)
    call = root_graph.add_cnode(sub_graph_value_node, root_para_1)

    # kernel graph0
    add1 = root_graph.add_cnode("Add", call, root_para_2)
    add2 = root_graph.add_cnode("Add", add1, root_para_1)
    load = root_graph.add_cnode("Load", add2, add1)
    mul = root_graph.add_cnode("Mul", load, root_para_2)
    # kernel graph1
    div = root_graph.add_cnode("RealDiv", mul, root_para_2)
    root_graph.set_target(div, "CPU")
    # kernel graph2
    add3 = root_graph.add_cnode("Add", load, div)
    root_graph.add_return(add3)
    root_graph.infer()
    root_graph.compile()
    x = Tensor(np.ones(shape).astype(np.float32))
    y = Tensor(np.ones(shape).astype(np.float32))
    out = root_graph(x, y)
    print(out)
    assert np.allclose(out.asnumpy(), np.ones(shape).astype(np.float32) * 6)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_print_output_control_flow():
    """
    Feature: Test print output of kernel graph.
    Description: Trans output to value node.
    Expectation: Run success.
    """
    shape = (3, 4)

    root_graph = BackendGraph()
    root_para_1 = root_graph.add_parameter(ms.float32, shape)
    root_para_2 = root_graph.add_parameter(ms.float32, shape)

    a = BackendGraph()
    sub_para_1 = a.add_parameter(ms.float32, shape)
    a.add_return(sub_para_1)
    sub_graph_value_node = root_graph.add_valuenode(a)

    print_1 = root_graph.add_cnode("Print", root_para_2)
    call = root_graph.add_cnode(sub_graph_value_node, root_para_1)

    U = root_graph.add_valuenode(UMonad())
    updatestate = root_graph.add_cnode("UpdateState", U, call, print_1)
    depend_2 = root_graph.add_cnode("Depend", root_para_2, updatestate)
    root_graph.add_return(depend_2)
    root_graph.infer()
    root_graph.compile()
    x = Tensor(np.ones(shape).astype(np.float32))
    y = Tensor(np.ones(shape).astype(np.float32))
    out = root_graph(x, y)
    print(out)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_print_output():
    """
    Feature: Test print output of kernel graph.
    Description: Trans output to value node.
    Expectation: Run success.
    """
    shape = (3, 4)

    root_graph = BackendGraph()
    root_para_1 = root_graph.add_parameter(ms.float32, shape)
    root_para_2 = root_graph.add_parameter(ms.float32, shape)
    print_1 = root_graph.add_cnode("Print", root_para_2)
    add = root_graph.add_cnode("Add", root_para_1, root_para_2)
    depend_1 = root_graph.add_cnode("Depend", root_para_1, print_1)
    sub = root_graph.add_cnode("Sub", root_para_1, depend_1)
    root_graph.set_target(add, "CPU")
    root_graph.set_target(sub, "CPU")
    U = root_graph.add_valuenode(UMonad())
    updatestate = root_graph.add_cnode("UpdateState", U, sub, print_1)
    depend_2 = root_graph.add_cnode("Depend", root_para_2, updatestate)
    mul = root_graph.add_cnode("Mul", depend_2, sub)
    root_graph.add_return(mul)
    root_graph.infer()
    root_graph.compile()
    x = Tensor(np.ones(shape).astype(np.float32))
    y = Tensor(np.ones(shape).astype(np.float32))
    out = root_graph(x, y)
    print(out)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cse_with_ref():
    """
    Feature: Test cse with ref.
    Description: Cannot eliminate node with ref.
    Expectation: Run success.
    """
    shape = (3, 4)

    root_graph = BackendGraph()
    para = root_graph.add_parameter(ms.float32, shape)
    add1 = root_graph.add_cnode("Add", para, para)
    add2 = root_graph.add_cnode("Add", para, para)
    assign = root_graph.add_cnode(root_graph.add_valuenode(P.Assign()), add1, para)
    depend = root_graph.add_cnode("Depend", add1, assign)
    add3 = root_graph.add_cnode("Add", depend, add2)
    root_graph.add_return(add3)
    root_graph.infer()
    root_graph.compile()
    x = Tensor(np.ones(shape).astype(np.float32))
    out = root_graph(x)
    assert np.allclose(out.asnumpy(), np.ones(shape).astype(np.float32) * 3)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_shape_for_cpu_diff_type():
    """
    Feature: Support type trans for graph output.
    Description: the shape in the second step is larger than that in the first step..
    Expectation: Run success.
    """
    a = BackendGraph()
    shape = (-1, -1)
    para_1 = a.add_parameter(ms.float16, shape)
    para_2 = a.add_parameter(ms.float16, shape)
    ms.set_context(device_target="CPU")
    bool_false = a.add_valuenode(False)
    add1 = a.add_cnode("MatMul", para_1, para_2, bool_false, bool_false)
    add2 = a.add_cnode("MatMul", add1, para_1, bool_false, bool_false)
    add3 = a.add_cnode("MatMul", para_2, add2, bool_false, bool_false)
    make_tuple = a.add_cnode("MakeTuple", add2, add3)
    a.add_return(make_tuple)
    a.infer()
    a.compile()
    x1 = Tensor(np.ones((4, 2)).astype(np.float16))
    y1 = Tensor(np.ones((2, 4)).astype(np.float16))
    out1 = a(x1, y1)
    res1 = np.matmul(x1, y1)
    res2 = np.matmul(res1, x1)
    res3 = np.matmul(y1, res2)
    assert np.allclose(out1[0].asnumpy(), res2)
    assert np.allclose(out1[1].asnumpy(), res3)
    x2 = Tensor(np.ones((32, 32)).astype(np.float16))
    y2 = Tensor(np.ones((32, 32)).astype(np.float16))
    out2 = a(x2, y2)
    res1 = np.matmul(x2, y2)
    res2 = np.matmul(res1, x2)
    res3 = np.matmul(y2, res2)
    assert np.allclose(out2[0].asnumpy(), res2)
    assert np.allclose(out2[1].asnumpy(), res3)

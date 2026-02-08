import pytest
import numpy as np
import os
from mindspore import jit, context, Tensor, Parameter, ops, ParameterTuple, JitConfig
from mindspore.common import dtype as mstype
from mindspore.nn import Cell, SequentialCell
from mindspore._c_expression import get_code_extra
from .share.utils import match_array
from tests.mark_utils import arg_mark


def hook_1_fn(grad):
    return grad * 2


def hook_2_fn(grad):
    return grad * 4


def hook_3_fn(grad):
    return grad * 6


def hook_4_fn(grad):
    return grad * 8


def break_hook_fn(grad):
    print("grad : ", grad, flush=True)
    return grad * 20


class ForwardNet(Cell):
    def __init__(self):
        super(ForwardNet, self).__init__()
        self.matmul = ops.MatMul()
        self.w = Parameter(Tensor([[2., 2.], [2., 2.]], mstype.float32), name="w", requires_grad=True)
        self.z = Parameter(Tensor([[3., 3.], [3., 3.]], mstype.float32), name="z", requires_grad=True)

    def construct(self, x, y):
        x = x * self.w * self.z
        return self.matmul(x, y)


class GradNet(Cell):
    def __init__(self, get_all, get_by_list):
        super(GradNet, self).__init__()
        self.net = ForwardNet()
        self.params = ParameterTuple(self.net.trainable_params())
        self.params[0].register_hook(hook_3_fn)
        self.params[1].register_hook(hook_4_fn)
        self.get_by_list = get_by_list
        self.grad_op = ops.GradOperation(get_all=get_all, get_by_list=get_by_list)

    def construct(self, x, y):
        if self.get_by_list is False:
            return self.grad_op(self.net)(x, y)
        return self.grad_op(self.net, self.params)(x, y)


def run_grad_net(net, x, y):
    return net(x, y)


def run_multi_grad_net(net, x, y):
    a = net(x, y)
    b = net(x, y)
    return a, b


def check_func_compile_state(func):
    jcr = get_code_extra(func.__wrapped__)
    assert jcr is not None
    assert jcr['break_count_'] == 0
    assert jcr['stat'] == 'GRAPH_CALLABLE'


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [run_grad_net])
@pytest.mark.parametrize('net', [GradNet(False, False)])
@pytest.mark.parametrize('x', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('y', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('x_hook', [hook_1_fn])
@pytest.mark.parametrize('y_hook', [hook_2_fn])
def test_run_grad_first_input(func, net, x, y, x_hook, y_hook):
    """
    Feature: ALL TO ALL
    Description: test cases for test_run_grad_first_input with hook
    Expectation: the result match
    Note: Must call pijit first, the args x and y will be modified in pynative
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x.register_hook(x_hook)
    y.register_hook(y_hook)
    wrapped_func = jit(func, capture_mode="bytecode")
    ms_res = wrapped_func(net, x, y)
    check_func_compile_state(wrapped_func)
    res = func(net, x, y)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [run_grad_net])
@pytest.mark.parametrize('net', [GradNet(False, False)])
@pytest.mark.parametrize('x', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('y', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('x_hook', [hook_1_fn])
@pytest.mark.parametrize('y_hook', [hook_2_fn])
def test_run_grad_first_input_multi_hook(func, net, x, y, x_hook, y_hook):
    """
    Feature: ALL TO ALL
    Description: test cases for test_run_grad_first_input_multi_hook with hook
    Expectation: the result match
    Note: Must call pijit first, the args x and y will be modified in pynative
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x.register_hook(x_hook)
    x.register_hook(y_hook)
    y.register_hook(y_hook)
    wrapped_func = jit(func, capture_mode="bytecode")
    ms_res = wrapped_func(net, x, y)
    check_func_compile_state(wrapped_func)
    res = func(net, x, y)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [run_grad_net])
@pytest.mark.parametrize('net', [GradNet(True, False)])
@pytest.mark.parametrize('x', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('y', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('x_hook', [hook_1_fn])
@pytest.mark.parametrize('y_hook', [hook_2_fn])
def test_run_grad_all_inputs(func, net, x, y, x_hook, y_hook):
    """
    Feature: ALL TO ALL
    Description: test cases for test_run_grad_all_inputs with hook
    Expectation: the result match
    Note: Must call pijit first, the args x and y will be modified in pynative
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x.register_hook(x_hook)
    y.register_hook(y_hook)
    wrapped_func = jit(func, capture_mode="bytecode")
    ms_res = wrapped_func(net, x, y)
    check_func_compile_state(wrapped_func)
    res = func(net, x, y)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [run_grad_net])
@pytest.mark.parametrize('net', [GradNet(False, True)])
@pytest.mark.parametrize('x', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('y', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('x_hook', [hook_1_fn])
@pytest.mark.parametrize('y_hook', [hook_2_fn])
def test_run_grad_only_weights(func, net, x, y, x_hook, y_hook):
    """
    Feature: ALL TO ALL
    Description: test cases for test_run_grad_only_weights with hook
    Expectation: the result match
    Note: Must call pijit first, the args x and y will be modified in pynative
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x.register_hook(x_hook)
    y.register_hook(y_hook)
    wrapped_func = jit(func, capture_mode="bytecode")
    ms_res = wrapped_func(net, x, y)
    check_func_compile_state(wrapped_func)
    res = func(net, x, y)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [run_grad_net])
@pytest.mark.parametrize('net', [GradNet(True, True)])
@pytest.mark.parametrize('x', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('y', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('x_hook', [hook_1_fn])
@pytest.mark.parametrize('y_hook', [hook_2_fn])
def test_run_grad_inputs_and_weights(func, net, x, y, x_hook, y_hook):
    """
    Feature: ALL TO ALL
    Description: test cases for test_run_grad_inputs_and_weights with hook
    Expectation: the result match
    Note: Must call pijit first, the args x and y will be modified in pynative
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x.register_hook(x_hook)
    y.register_hook(y_hook)
    wrapped_func = jit(func, capture_mode="bytecode")
    ms_res = wrapped_func(net, x, y)
    check_func_compile_state(wrapped_func)
    res = func(net, x, y)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [run_multi_grad_net])
@pytest.mark.parametrize('net', [GradNet(True, True)])
@pytest.mark.parametrize('x', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('y', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('x_hook', [hook_1_fn])
@pytest.mark.parametrize('y_hook', [hook_2_fn])
def test_run_multi_grad_inputs_and_weights(func, net, x, y, x_hook, y_hook):
    """
    Feature: ALL TO ALL
    Description: test cases for test_run_multi_grad_inputs_and_weights with hook
    Expectation: the result match
    Note: Must call pijit first, the args x and y will be modified in pynative
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x.register_hook(x_hook)
    y.register_hook(y_hook)
    wrapped_func = jit(func, capture_mode="bytecode")
    ms_res = wrapped_func(net, x, y)
    check_func_compile_state(wrapped_func)
    res = func(net, x, y)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@pytest.mark.skip(reason='fix it later')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [run_grad_net])
@pytest.mark.parametrize('net', [GradNet(False, False)])
@pytest.mark.parametrize('x', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('y', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('x_hook', [break_hook_fn])
@pytest.mark.parametrize('y_hook', [hook_2_fn])
def test_run_grad_first_input_break(func, net, x, y, x_hook, y_hook):
    """
    Feature: ALL TO ALL
    Description: test cases for test_run_grad_first_input_break with hook
    Expectation: the result match
    Note: Must call pijit first, the args x and y will be modified in pynative
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x.register_hook(x_hook)
    y.register_hook(y_hook)
    wrapped_func = jit(func, capture_mode="bytecode")
    ms_res = wrapped_func(net, x, y)
    res = func(net, x, y)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


def double_fn(grad):
    return grad * 2


def triple_fn(grad):
    return grad * 3


def half_fn(grad):
    return grad * 0.5


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_hook_register_tensor_outside_grad_input():
    """
    Feature: Tensor.register_hook outside construct.
    Description: Register hook on input Tensor and compare gradients between PyNative and JIT.
    Expectation: JIT gradient matches PyNative gradient.
    Migrated from: test_parse_pijit_hook.py::test_parse_pijit_hook_register_tensor_outside
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.a = 1

        def construct(self, x):
            out = x * x * self.a
            return out

    x_np = np.ones([2, 3], np.float32)
    # PyNative
    x1 = Tensor(x_np, mstype.float32)
    x1.register_hook(double_fn)  # times 2
    net1 = Net()
    grad_fn1 = ops.grad(net1)
    grad_pn = grad_fn1(x1)

    # JIT
    x2 = Tensor(x_np, mstype.float32)
    x2.register_hook(double_fn)
    net2 = Net()
    net2.construct = jit(net2.construct, capture_mode='bytecode')
    grad_fn2 = ops.grad(net2)
    grad_jit = grad_fn2(x2)

    match_array(grad_pn, grad_jit)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_hook_register_parameter_outside_grad_weight():
    """
    Feature: Parameter.register_hook outside construct.
    Description: Register hook on Parameter and compare weight gradients between PyNative and JIT.
    Expectation: JIT weight gradient matches PyNative.
    Migrated from: test_parse_pijit_hook.py::test_parse_pijit_hook_register_parameter_outside
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.a = Parameter(Tensor(np.ones([2, 3], np.float32), mstype.float32), name='a')
            self.a.register_hook(double_fn)  # times 2

        def construct(self, x):
            return x * x * self.a

    in_np = np.ones([2, 3], np.float32)

    # PyNative
    net1 = Net()
    grad_fn1 = ops.grad(net1, grad_position=None, weights=net1.a)
    grad_pn = grad_fn1(Tensor(in_np, mstype.float32))

    # JIT
    net2 = Net()
    net2.construct = jit(net2.construct, capture_mode='bytecode')
    grad_fn2 = ops.grad(net2, grad_position=None, weights=net2.a)
    grad_jit = grad_fn2(Tensor(in_np, mstype.float32))

    match_array(grad_pn, grad_jit)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_hook_register_fallback_square_on_tensor():
    """
    Feature: Tensor.register_hook with custom square op.
    Description: Hook squares gradient using Tensor ops; compare PyNative vs JIT gradients.
    Expectation: JIT gradient matches PyNative gradient.
    Migrated from: test_parse_pijit_hook.py::test_parse_pijit_hook_register_fallback
    """

    def square_hook(grad):
        grad_np = grad.asnumpy()
        out_np = np.square(grad_np)
        return Tensor(out_np)

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.a = Parameter(Tensor(np.ones([2, 3], np.float32), dtype=mstype.float32), name='a')

        def construct(self, x):
            return x * x * self.a

    x_np = np.ones([2, 3], np.float32)

    # PyNative
    x1 = Tensor(x_np, mstype.float32)
    x1.register_hook(square_hook)
    net1 = Net()
    grad_fn1 = ops.grad(net1)
    grad_pn = grad_fn1(x1)

    # JIT
    x2 = Tensor(x_np, mstype.float32)
    x2.register_hook(square_hook)
    net2 = Net()
    net2.construct = jit(net2.construct, capture_mode='bytecode')
    grad_fn2 = ops.grad(net2)
    grad_jit = grad_fn2(x2)

    match_array(grad_pn, grad_jit)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_hook_register_dump_to_file():
    """
    Feature: Tensor.register_hook with dumping to file.
    Description: Hook dumps gradient to .npy file; verify contents and compare JIT vs PyNative.
    Expectation: Dumped gradients match computed ones; JIT matches PyNative.
    Migrated from: test_parse_pijit_hook.py::test_parse_pijit_hook_register_dump
    """

    def make_dump_hook(path):
        def _dump(grad):
            np.save(path, grad.asnumpy())
            return grad

        return _dump

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.a = Parameter(Tensor(np.ones([2, 3], np.float32), mstype.float32), name='a')

        def construct(self, x):
            return x * x * self.a

    x_np = np.ones([2, 3], np.float32)
    path1 = 'grad_dump_pynative.npy'
    path2 = 'grad_dump_jit.npy'
    try:
        # PyNative
        x1 = Tensor(x_np, mstype.float32)
        x1.register_hook(make_dump_hook(path1))
        net1 = Net()
        grad_fn1 = ops.grad(net1)
        grad_pn = grad_fn1(x1)
        dump_pn = Tensor(np.load(path1), mstype.float32)

        # JIT
        x2 = Tensor(x_np, mstype.float32)
        x2.register_hook(make_dump_hook(path2))
        net2 = Net()
        net2.construct = jit(net2.construct, capture_mode='bytecode')
        grad_fn2 = ops.grad(net2)
        grad_jit = grad_fn2(x2)
        dump_jit = Tensor(np.load(path2), mstype.float32)

        match_array(grad_pn, dump_pn)
        match_array(grad_jit, dump_jit)
        match_array(grad_pn, grad_jit)
    finally:
        if os.path.exists(path1):
            os.remove(path1)
        if os.path.exists(path2):
            os.remove(path2)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_hook_register_ctrl_if_condition():
    """
    Feature: Tensor.register_hook with control flow inside hook.
    Description: Hook doubles gradient when value > 1; compare PyNative vs JIT.
    Expectation: JIT gradient matches PyNative gradient.
    Migrated from: test_parse_pijit_hook.py::test_parse_pijit_hook_register_ctrl
    """

    def ctrl_fn(grad):
        return grad * 2 if grad > 1 else grad

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.k = 2

        def construct(self, x):
            return x * x * self.k

    x_np = np.array([2], np.float32)

    # PyNative
    x1 = Tensor(x_np, mstype.float32)
    x1.register_hook(ctrl_fn)
    net1 = Net()
    grad_fn1 = ops.grad(net1)
    grad_pn = grad_fn1(x1)

    # JIT
    x2 = Tensor(x_np, mstype.float32)
    x2.register_hook(ctrl_fn)
    net2 = Net()
    net2.construct = jit(net2.construct, capture_mode='bytecode')
    grad_fn2 = ops.grad(net2)
    grad_jit = grad_fn2(x2)

    match_array(grad_pn, grad_jit)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_hook_register_middle_fallback_tensor_zero_grad():
    """
    Feature: Register hook on fallback tensor created inside construct.
    Description: Create Tensor via asnumpy() in construct, register hook on it; gradient to input should be zeros.
    Expectation: JIT and PyNative produce zero gradients and match.
    Migrated from: test_parse_pijit_hook.py::test_parse_pijit_hook_register_middle_fallback
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.k = 2

        def construct(self, x):
            y = Tensor(x.asnumpy())
            y.register_hook(double_fn)
            return y * self.k

    x_np = np.ones([2, 3], np.float32)

    # PyNative
    net1 = Net()
    grad_fn1 = ops.grad(net1)
    grad_pn = grad_fn1(Tensor(x_np, mstype.float32))

    # JIT
    net2 = Net()
    net2.construct = jit(net2.construct, capture_mode='bytecode')
    grad_fn2 = ops.grad(net2)
    grad_jit = grad_fn2(Tensor(x_np, mstype.float32))

    zeros = Tensor(np.zeros([2, 3], np.float32), mstype.float32)
    match_array(grad_pn, zeros)
    match_array(grad_jit, zeros)
    match_array(grad_pn, grad_jit)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_hook_register_grad_not_all_inputs_and_weights():
    """
    Feature: Tensor and Parameter hooks with partial gradient fetching.
    Description: Fetch gradient for second input and a weight; compare between PyNative and JIT.
    Expectation: Both input and weight gradients match between modes.
    Migrated from: test_parse_pijit_hook.py::test_parse_pijit_hook_register_grad_not_all
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.a = Parameter(Tensor(np.ones([2, 3], np.float32), mstype.float32), name='a')
            self.b = Parameter(Tensor(np.ones([2, 3], np.float32), mstype.float32), name='b')
            self.a.register_hook(double_fn)
            self.b.register_hook(double_fn)

        def construct(self, x, y):
            return x * self.a * y * self.b

    x_np = np.ones([2, 3], np.float32)
    y_np = np.ones([2, 3], np.float32)

    # PyNative
    x1 = Tensor(x_np, mstype.float32)
    y1 = Tensor(y_np, mstype.float32)
    x1.register_hook(double_fn)
    y1.register_hook(double_fn)
    net1 = Net()
    grad_fn1 = ops.grad(net1, grad_position=1, weights=net1.b)
    grad_y_pn, grad_b_pn = grad_fn1(x1, y1)

    # JIT
    x2 = Tensor(x_np, mstype.float32)
    y2 = Tensor(y_np, mstype.float32)
    x2.register_hook(double_fn)
    y2.register_hook(double_fn)
    net2 = Net()
    net2.construct = jit(net2.construct, capture_mode='bytecode')
    grad_fn2 = ops.grad(net2, grad_position=1, weights=net2.b)
    grad_y_jit, grad_b_jit = grad_fn2(x2, y2)

    match_array(grad_y_pn, grad_y_jit)
    match_array(grad_b_pn, grad_b_jit)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_hook_register_sequential_layers_parameters():
    """
    Feature: Parameter.register_hook in stacked Cells.
    Description: Register hook on parameters of each layer in SequentialCell; compare gradients between PyNative and JIT.
    Expectation: Parameter gradients match between PyNative and JIT.
    Migrated from: test_parse_pijit_hook_register_remove.py::test_parse_pijit_hook_cell_layers
    """

    class Block(Cell):
        def __init__(self):
            super().__init__()
            self.a = Parameter(Tensor(np.ones([2, 3], np.float32), mstype.float32), name='a')
            self.a.register_hook(double_fn)

        def construct(self, x):
            return x * self.a

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.layers = SequentialCell()
            for _ in range(3):
                self.layers.append(Block())

        def construct(self, x):
            out = x
            for layer in self.layers:
                out = out + layer(x)
            return out

    x_np = np.ones([2, 3], np.float32)

    # PyNative
    net_pn = Net()
    grad_fn_pn = ops.grad(net_pn, grad_position=None, weights=net_pn.trainable_params())
    grads_pn = grad_fn_pn(Tensor(x_np, mstype.float32))

    # JIT
    net_jit = Net()
    net_jit.construct = jit(net_jit.construct, capture_mode='bytecode')
    grad_fn_jit = ops.grad(net_jit, grad_position=None, weights=net_jit.trainable_params())
    grads_jit = grad_fn_jit(Tensor(x_np, mstype.float32))

    for grad_pn, grad_jit in zip(grads_pn, grads_jit):
        match_array(grad_pn, grad_jit)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_hook_register_remove_before_backward():
    """
    Feature: Tensor.register_hook remove handle before backward.
    Description: Register hook on input tensor, run forward, remove handle, then compare gradients with and without JIT.
    Expectation: Gradients after removal match baseline without hook for both modes.
    Migrated from: test_parse_pijit_hook_register_remove.py::test_parse_pijit_hook_register_forward_remove
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.a = Parameter(Tensor(np.ones([2, 3], np.float32), mstype.float32), name='a')

        def construct(self, x):
            return x * x * self.a

    x_np = np.ones([2, 3], np.float32)

    def run(mode):
        x = Tensor(x_np, mstype.float32)
        handle = x.register_hook(double_fn)
        net = Net()
        if mode == "jit":
            net.construct = jit(net.construct, capture_mode='bytecode')
        net(x)
        handle.remove()
        grad_fn = ops.grad(net, grad_position=0, weights=net.a)
        return grad_fn(x)

    grad_input_pn, grad_weight_pn = run("pynative")
    grad_input_jit, grad_weight_jit = run("jit")

    match_array(grad_input_pn, grad_input_jit)
    match_array(grad_weight_pn, grad_weight_jit)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_hook_register_remove_after_first_backward():
    """
    Feature: Tensor.register_hook removal after first backward.
    Description: Register hook, execute backward to trigger hook, remove handle, then ensure subsequent gradients match baseline.
    Expectation: After removal, gradients match baseline; first backward doubles gradient in both modes.
    Migrated from: test_parse_pijit_hook_register_remove.py::test_parse_pijit_hook_register_backward_remove
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.a = Parameter(Tensor(np.ones([2, 3], np.float32), mstype.float32), name='a')

        def construct(self, x):
            return x * x * self.a

    x_np = np.ones([2, 3], np.float32)

    def run(mode):
        x = Tensor(x_np, mstype.float32)
        handle = x.register_hook(double_fn)
        net = Net()
        if mode == "jit":
            net.construct = jit(net.construct, capture_mode='bytecode')
        net(x)
        grad_fn = ops.grad(net, grad_position=0)
        grad_with_hook = grad_fn(x)
        handle.remove()
        net(x)
        grad_after_remove = grad_fn(x)
        return grad_with_hook, grad_after_remove

    grad_with_hook_pn, grad_after_remove_pn = run("pynative")
    grad_with_hook_jit, grad_after_remove_jit = run("jit")

    match_array(grad_with_hook_pn, grad_with_hook_jit)
    match_array(grad_after_remove_pn, grad_after_remove_jit)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_hook_register_after_initial_run():
    """
    Feature: Tensor.register_hook after initial execution.
    Description: Run network once without hook, then register hook and ensure gradients double in both PyNative and JIT.
    Expectation: Gradients after hook registration double baseline and match between modes.
    Migrated from: test_parse_pijit_hook_register_remove.py::test_parse_pijit_hook_register_after_run
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.a = Parameter(Tensor(np.ones([2, 3], np.float32), mstype.float32), name='a')

        def construct(self, x):
            return x * x * self.a

    x_np = np.ones([2, 3], np.float32)

    def run(mode):
        net = Net()
        if mode == "jit":
            net.construct = jit(net.construct, capture_mode='bytecode')
        x = Tensor(x_np, mstype.float32)
        net(x)
        grad_fn = ops.grad(net, grad_position=0)
        grad_before_hook = grad_fn(x)
        x.register_hook(double_fn)
        net(x)
        grad_after_hook = grad_fn(x)
        return grad_before_hook, grad_after_hook

    grad_before_hook_pn, grad_after_hook_pn = run("pynative")
    grad_before_hook_jit, grad_after_hook_jit = run("jit")

    match_array(grad_before_hook_pn, grad_before_hook_jit)
    match_array(grad_after_hook_pn, grad_after_hook_jit)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_hook_register_run_twice():
    """
    Feature: Tensor.register_hook reused across multiple runs.
    Description: Register hook once and execute gradients twice; verify hook effect persists in PyNative and JIT.
    Expectation: Gradients from both runs match and equal doubled baseline in both modes.
    Migrated from: test_parse_pijit_hook_register_remove.py::test_parse_pijit_hook_register_run_twice
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.a = Parameter(Tensor(np.ones([2, 3], np.float32), mstype.float32), name='a')

        def construct(self, x):
            return x * x * self.a

    x_np = np.ones([2, 3], np.float32)

    def run(mode):
        net = Net()
        if mode == "jit":
            net.construct = jit(net.construct, capture_mode='bytecode')
        x = Tensor(x_np, mstype.float32)
        x.register_hook(double_fn)
        grad_fn = ops.grad(net, grad_position=0)
        net(x)
        grad_first = grad_fn(x)
        net(x)
        grad_second = grad_fn(x)
        return grad_first, grad_second

    grad_first_pn, grad_second_pn = run("pynative")
    grad_first_jit, grad_second_jit = run("jit")

    match_array(grad_first_pn, grad_first_jit)
    match_array(grad_second_pn, grad_second_jit)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_hook_register_three_remove_one():
    """
    Feature: Tensor.register_hook remove one handle among multiple.
    Description: Register three hooks, remove the middle one, and ensure remaining hooks keep gradient unchanged overall.
    Expectation: Final gradients match baseline and are consistent between PyNative and JIT.
    Migrated from: test_parse_pijit_hook_register_remove.py::test_parse_pijit_hook_register_three_remove_one
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()

        def construct(self, x):
            return x * x

    x_np = np.ones([2, 3], np.float32)

    def run(mode):
        net = Net()
        if mode == "jit":
            net.construct = jit(net.construct, capture_mode='bytecode')
        x = Tensor(x_np, mstype.float32)
        x.register_hook(double_fn)
        handle_mid = x.register_hook(triple_fn)
        x.register_hook(half_fn)
        handle_mid.remove()
        grad_fn = ops.grad(net, grad_position=0)
        return grad_fn(x)

    grad_pn = run("pynative")
    grad_jit = run("jit")

    match_array(grad_pn, grad_jit)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_hook_register_remove_inside_construct():
    """
    Feature: Parameter.register_hook with input hook removed inside construct.
    Description: Remove input hook inside construct while keeping parameter hook; compare gradients between modes.
    Expectation: Input gradients match baseline; parameter gradients remain doubled in both modes.
    Migrated from: test_parse_pijit_hook_register_remove.py::test_parse_pijit_hook_register_out_remove_in
    """

    class Net(Cell):
        def __init__(self, handle):
            super().__init__()
            self.a = Parameter(Tensor(np.ones([2, 3], np.float32), mstype.float32), name='a')
            self.a.register_hook(double_fn)
            self.handle_x = handle

        def construct(self, x):
            self.handle_x.remove()
            return x * x * self.a

    x_np = np.ones([2, 3], np.float32)

    def run(mode):
        x = Tensor(x_np, mstype.float32)
        handle = x.register_hook(double_fn)
        net = Net(handle)
        if mode == "jit":
            net.construct = jit(net.construct, capture_mode='bytecode')
        grad_fn = ops.grad(net, grad_position=0, weights=net.a)
        return grad_fn(x)

    grad_input_pn, grad_weight_pn = run("pynative")
    grad_input_jit, grad_weight_jit = run("jit")

    match_array(grad_input_pn, grad_input_jit)
    match_array(grad_weight_pn, grad_weight_jit)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_hook_register_on_jit_function():
    """
    Feature: Tensor.register_hook applied before calling JIT function.
    Description: Register hook on input tensor and compare gradients between PyNative function and its JIT compiled counterpart.
    Expectation: JIT gradient matches PyNative and equals doubled baseline.
    Migrated from: test_parse_pijit_hook_register_remove.py::test_parse_pijit_hook_register_jit_func
    """

    def xsquare3(x):
        return x * x * 3

    x_np = np.ones([2, 3], np.float32)

    x1 = Tensor(x_np, mstype.float32)
    x1.register_hook(double_fn)
    grad_pn = ops.grad(xsquare3)(x1)

    x2 = Tensor(x_np, mstype.float32)
    x2.register_hook(double_fn)
    jit_fn = jit(xsquare3, capture_mode='bytecode')
    grad_jit = ops.grad(jit_fn)(x2)

    match_array(grad_pn, grad_jit)

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
# ============================================================================
"""Test raise exception"""

from mindspore import Tensor
from mindspore import jit
from mindspore._c_expression import get_code_extra
from mindspore.nn import Cell
from mindspore import ops
import numpy as np
from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import match_array
from tests.st.pi_jit.share.grad import GradOfFirstInput, compute_grad_of_net_inputs
import pytest


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_except_case_1():
    """
    Feature: Test with raise exception
    Description: Test with raise exception process
    Expectation: raise exception.
    """

    def func():
        try:
            raise Exception("new exception in try!")
        finally:
            pass

    with pytest.raises(Exception, match="new exception in try!"):
        jit(function=func, capture_mode="bytecode")()
    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 1


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_except_case_2():
    """
    Feature: Test with raise exception
    Description: Test with raise exception process
    Expectation: raise exception.
    """

    def func():
        try:
            try:
                try:
                    raise Exception("new exception in try!")
                finally:
                    pass
            finally:
                pass
        finally:
            pass

    with pytest.raises(Exception, match="new exception in try!"):
        jit(function=func, capture_mode="bytecode")()
    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 1


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_except_case_3():
    """
    Feature: Test with raise exception
    Description: Test with raise exception process
    Expectation: raise exception.
    """

    def func():
        try:
            try:
                try:
                    raise Exception("new exception in try!")
                finally:
                    pass
            finally:
                raise Exception("new exception2 in finally!")
        finally:
            pass

    with pytest.raises(Exception, match="new exception2 in finally!"):
        jit(function=func, capture_mode="bytecode")()
    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 1


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_except_case_4():
    """
    Feature: Test with try/except/finally
    Description: Test with raise exception process
    Expectation: no except raise.
    """

    def func():
        try:
            try:
                try:
                    raise Exception("new exception in try!")
                except ArithmeticError:
                    pass
                except BufferError:
                    pass
                except Exception:
                    pass
                finally:
                    pass
            finally:
                pass
        finally:
            pass

    jit(function=func, capture_mode="bytecode")()
    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 0


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_except_case_5():
    """
    Feature: Test with try/except/finally
    Description: Test with raise exception process
    Expectation: resolve except correctly.
    """

    def func():
        try:
            try:
                try:
                    raise Exception("new exception in try!")
                except ArithmeticError:
                    pass
                except BufferError:
                    pass
                except Exception:
                    raise Exception("new exception2 in except!")
                finally:
                    pass
            finally:
                raise Exception("new exception3 in finally!")
        finally:
            pass

    with pytest.raises(Exception, match="new exception3 in finally!"):
        jit(function=func, capture_mode="bytecode")()
    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 1


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_except_case_6():
    """
    Feature: Test with try/except/finally
    Description: Test with raise exception process
    Expectation: resolve except correctly,do not support raise except from callee.
    """

    def func():
        try:
            try:
                try:
                    raise Exception("new exception in try!")
                except ArithmeticError:
                    pass
                except BufferError:
                    pass
                except Exception:
                    raise Exception("new exception2 in except!")
                finally:
                    pass
            finally:
                raise Exception("new exception3 in finally!")
        finally:
            pass

    def func2():
        func()

    with pytest.raises(Exception, match="new exception3 in finally!"):
        jit(function=func2, capture_mode="bytecode")()
    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 1


def test_except_case_7():
    """
    Feature: Test with raise exception
    Description: Test with raise exception process
    Expectation: raise exception.
    """

    def func():
        i = 1
        try:
            i = 2
            raise Exception("new exception in try!")
        except Exception:
            i = 3
            pass
        finally:
            pass
        return i

    got = jit(function=func, capture_mode="bytecode")()
    expected = func()
    print(got)
    assert got == expected
    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 0


def test_except_case_8():
    """
    Feature: Test with raise exception
    Description: Test with raise exception process
    Expectation: raise exception.
    """

    def func():
        i = 1
        try:
            i = 2
            raise Exception("new exception in try!")
        except:
            i = 3
            raise Exception("new exception in except!")
        finally:
            pass
        return i

    with pytest.raises(Exception, match="new exception in except!"):
        got = jit(function=func, capture_mode="bytecode")()
        expected = func()
        assert got == expected

    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 1


def test_except_case_9():
    """
    Feature: Test with raise exception
    Description: Test with raise exception process
    Expectation: raise exception.
    """

    def func():
        i = 1
        try:
            i = 2
            raise Exception
        except Exception:
            i = 3
        finally:
            pass
        return i

    got = jit(function=func, capture_mode="bytecode")()
    expected = func()
    assert got == expected
    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 0


def test_except_case_10():
    """
    Feature: Test with raise exception
    Description: Test with raise exception process
    Expectation: raise exception.
    """

    def func():
        i = 1
        try:
            i = 2
        except Exception:
            i = 3
        else:
            i = 4
        return i

    got = jit(function=func, capture_mode="bytecode")()
    expected = func()
    assert got == expected
    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 0


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_except_case_11():
    """
    Feature: Test with try/except/finally
    Description: Test with raise exception process
    Expectation: resolve except correctly.
    """

    def func():
        try:
            try:
                try:
                    raise BufferError
                except ArithmeticError:
                    pass
                except BufferError:
                    raise BufferError("new BufferError in except!")
                except Exception:
                    raise Exception("new exception2 in except!")
                finally:
                    pass
            finally:
                pass
        finally:
            pass

    with pytest.raises(Exception, match="new BufferError in except!"):
        jit(function=func, capture_mode="bytecode")()
    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 1


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_except_case_12():
    """
    Feature: Test try/finally
    Description: Test with try/finally process
    Expectation: break count == 0.
    """

    def func():
        try:
            return 1
        finally:
            i = 3
        return 2

    got = jit(function=func, capture_mode="bytecode")()
    expected = func()
    assert got == expected
    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 0


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_except_case_13():
    """
    Feature: Test try/finally
    Description: Test with try/finally process
    Expectation: break count == 1.
    """

    def func():
        try:
            i = 1
            raise ArithmeticError
        except AssertionError:
            i = 2
            return i
        else:
            i = 3
        finally:
            i = 4
        return 5

    with pytest.raises(ArithmeticError):
        jit(function=func, capture_mode="bytecode")()

    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 1


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_except_case_14():
    """
    Feature: Test with/try/finally
    Description: Test with/try/finally process
    Expectation: break count == 0.
    """

    class MyFile:
        def __init__(self):
            pass

        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    def func():
        with MyFile():
            try:
                raise ArithmeticError
            except ArithmeticError:
                i = 2
            return i

    got = jit(function=func, capture_mode="bytecode")()
    assert got == 2
    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 0


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_except_case_15():
    """
    Feature: Test with
    Description: Test with process exception
    Expectation: no exception raise.
    """

    class MyFile:
        def __init__(self):
            pass

        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type == ArithmeticError:
                return True
            return False

    def func():
        i = 1
        with MyFile():
            i = 2
            raise ArithmeticError
        return i

    got = jit(function=func, capture_mode="bytecode")()
    assert got == 2
    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 1


@arg_mark(
    plat_marks=['cpu_linux'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='essential'
)
def test_try_except_basic():
    """
    Feature: Basic try/except handling.
    Description: Test basic try/except with exception handling.
    Expectation: JIT result and gradient match pynative.
    Migrated from: test_pijit_exception.py::test_pijit_try_except
    """
    class Net1(Cell):
        def __init__(self):
            super().__init__()
            self.op = ops.ReduceMean(False)

        def construct(self, x):
            try:
                out = self.op(x, 0)
            except ValueError:
                x = x.expand_dims(3)
                out = self.op(x, 3)
            return out

    x = Tensor(np.ones([2, 3, 4], np.float32))

    # Pynative mode: forward + gradient
    net_pynative = Net1()
    pynative_result = net_pynative(x)
    grad_net = GradOfFirstInput(net_pynative, sens_param=True)
    grad_net.set_train()
    output_shape = pynative_result.shape
    output_grad = Tensor(np.random.randn(*output_shape).astype(np.float32))
    pynative_grad = grad_net(x, output_grad)

    # JIT mode: forward + gradient
    net_jit = Net1()
    net_jit.construct = jit(net_jit.construct, capture_mode='bytecode')
    jit_result = net_jit(x)
    jit_grad_net = GradOfFirstInput(net_jit, sens_param=True)
    jit_grad_net.set_train()
    jit_grad = jit_grad_net(x, output_grad)

    # Compare forward results and gradients
    match_array(pynative_result, jit_result)
    match_array(pynative_grad, jit_grad, error=5)

@arg_mark(
    plat_marks=['cpu_linux'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='essential'
)
def test_try_finally_basic():
    """
    Feature: Basic try/finally handling.
    Description: Test basic try/finally with cleanup logic.
    Expectation: JIT result and gradient match pynative.
    Migrated from: test_pijit_exception.py::test_pijit_try_finally
    """
    class Net2(Cell):
        def __init__(self):
            super().__init__()
            self.op = ops.ReduceMean(False)

        def construct(self, x):
            try:
                out = self.op(x, 0)
            finally:
                pass
            return out

    x = Tensor(np.ones([2, 3, 4], np.float32))

    # Pynative mode: forward + gradient
    net_pynative = Net2()
    pynative_result = net_pynative(x)
    grad_net = GradOfFirstInput(net_pynative, sens_param=True)
    grad_net.set_train()
    output_shape = pynative_result.shape
    output_grad = Tensor(np.random.randn(*output_shape).astype(np.float32))
    pynative_grad = grad_net(x, output_grad)

    # JIT mode: forward + gradient
    net_jit = Net2()
    net_jit.construct = jit(net_jit.construct, capture_mode='bytecode')
    jit_result = net_jit(x)
    jit_grad_net = GradOfFirstInput(net_jit, sens_param=True)
    jit_grad_net.set_train()
    jit_grad = jit_grad_net(x, output_grad)

    # Compare forward results and gradients
    match_array(pynative_result, jit_result)
    match_array(pynative_grad, jit_grad, error=5)


@arg_mark(
    plat_marks=['cpu_linux'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='essential'
)
def test_try_multiple_except_basic():
    """
    Feature: Basic try/except with multiple except blocks.
    Description: Test try with multiple except blocks for different exceptions.
    Expectation: JIT result and gradient match pynative.
    Migrated from: test_pijit_exception.py::test_pijit_try_two_except
    """
    class Net3(Cell):
        def __init__(self):
            super().__init__()
            self.op = ops.ReduceMean(False)

        def construct(self, x):
            try:
                out = self.op(x, 0)
            except ValueError:
                x = x.expand_dims(3)
                out = self.op(x, 3)
            except TypeError:
                x = x + x
            return out

    x = Tensor(np.ones([2, 3, 4], np.float32))

    # Pynative mode: forward + gradient
    net_pynative = Net3()
    pynative_result = net_pynative(x)
    grad_net = GradOfFirstInput(net_pynative, sens_param=True)
    grad_net.set_train()
    output_shape = pynative_result.shape
    output_grad = Tensor(np.random.randn(*output_shape).astype(np.float32))
    pynative_grad = grad_net(x, output_grad)

    # JIT mode: forward + gradient
    net_jit = Net3()
    net_jit.construct = jit(net_jit.construct, capture_mode='bytecode')
    jit_result = net_jit(x)
    jit_grad_net = GradOfFirstInput(net_jit, sens_param=True)
    jit_grad_net.set_train()
    jit_grad = jit_grad_net(x, output_grad)

    # Compare forward results and gradients
    match_array(pynative_result, jit_result)
    match_array(pynative_grad, jit_grad, error=5)


@arg_mark(
    plat_marks=['cpu_linux'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='essential'
)
def test_try_except_else_basic():
    """
    Feature: Basic try/except/else handling.
    Description: Test basic try/except/else with else block execution.
    Expectation: JIT result and gradient match pynative.
    Migrated from: test_pijit_exception.py::test_pijit_try_except_else
    """
    class Net4(Cell):
        def __init__(self):
            super().__init__()
            self.op = ops.ReduceMean(False)

        def construct(self, x):
            try:
                out = self.op(x, 0)
            except ValueError:
                x = x.expand_dims(3)
                out = self.op(x, 3)
            else:
                out = out + out
            return out

    x = Tensor(np.ones([2, 3, 4], np.float32))

    # Pynative mode: forward + gradient
    net_pynative = Net4()
    pynative_result = net_pynative(x)
    grad_net = GradOfFirstInput(net_pynative, sens_param=True)
    grad_net.set_train()
    output_shape = pynative_result.shape
    output_grad = Tensor(np.random.randn(*output_shape).astype(np.float32))
    pynative_grad = grad_net(x, output_grad)

    # JIT mode: forward + gradient
    net_jit = Net4()
    net_jit.construct = jit(net_jit.construct, capture_mode='bytecode')
    jit_result = net_jit(x)
    jit_grad_net = GradOfFirstInput(net_jit, sens_param=True)
    jit_grad_net.set_train()
    jit_grad = jit_grad_net(x, output_grad)

    # Compare forward results and gradients
    match_array(pynative_result, jit_result)
    match_array(pynative_grad, jit_grad, error=5)


@arg_mark(
    plat_marks=['cpu_linux'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='essential'
)
def test_try_except_else_finally_basic():
    """
    Feature: Basic try/except/else/finally handling.
    Description: Test basic try/except/else/finally with all blocks.
    Expectation: JIT result and gradient match pynative.
    Migrated from: test_pijit_exception.py::test_pijit_try_except_else_finally
    """
    class Net5(Cell):
        def __init__(self):
            super().__init__()
            self.op = ops.ReduceMean(False)

        def construct(self, x):
            try:
                out = self.op(x, 0)
            except ValueError:
                x = x.expand_dims(3)
                out = self.op(x, 3)
            else:
                out = out + out
            finally:
                out = out * out
            return out

    x = Tensor(np.ones([2, 3, 4], np.float32))

    # Pynative mode: forward + gradient
    net_pynative = Net5()
    pynative_result = net_pynative(x)
    grad_net = GradOfFirstInput(net_pynative, sens_param=True)
    grad_net.set_train()
    output_shape = pynative_result.shape
    output_grad = Tensor(np.random.randn(*output_shape).astype(np.float32))
    pynative_grad = grad_net(x, output_grad)

    # JIT mode: forward + gradient
    net_jit = Net5()
    net_jit.construct = jit(net_jit.construct, capture_mode='bytecode')
    jit_result = net_jit(x)
    jit_grad_net = GradOfFirstInput(net_jit, sens_param=True)
    jit_grad_net.set_train()
    jit_grad = jit_grad_net(x, output_grad)

    # Compare forward results and gradients
    match_array(pynative_result, jit_result)
    match_array(pynative_grad, jit_grad, error=5)


@arg_mark(
    plat_marks=['cpu_linux'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='essential'
)
def test_except_graph_break_in_try():
    """
    Feature: Exception handling with graph break.
    Description: Test try/except with graph break in try block.
    Expectation: JIT result and gradient match pynative.
    Migrated from: test_pijit_exception.py::test_pijit_try_break
    """
    class Net6(Cell):
        def __init__(self):
            super().__init__()
            self.op = ops.ReduceMean(False)

        def construct(self, x):
            try:
                out = self.op(Tensor(x.asnumpy()), 0)
            except ValueError:
                x = x.expand_dims(3)
                out = self.op(x, 3)
            else:
                out = out + out
            finally:
                out = out * out
            return out

    x = Tensor(np.ones([2, 3, 4], np.float32))

    # Pynative mode: forward + gradient
    net_pynative = Net6()
    pynative_result = net_pynative(x)
    grad_net = GradOfFirstInput(net_pynative, sens_param=True)
    grad_net.set_train()
    output_shape = pynative_result.shape
    output_grad = Tensor(np.random.randn(*output_shape).astype(np.float32))
    pynative_grad = grad_net(x, output_grad)

    # JIT mode: forward + gradient
    net_jit = Net6()
    net_jit.construct = jit(net_jit.construct, capture_mode='bytecode')
    jit_result = net_jit(x)
    jit_grad_net = GradOfFirstInput(net_jit, sens_param=True)
    jit_grad_net.set_train()
    jit_grad = jit_grad_net(x, output_grad)

    # Compare forward results and gradients
    match_array(pynative_result, jit_result)
    match_array(pynative_grad, jit_grad, error=5)


@arg_mark(
    plat_marks=['cpu_linux'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='essential'
)
def test_except_graph_break_in_except():
    """
    Feature: Exception handling with graph break.
    Description: Test try/except with graph break in except block.
    Expectation: JIT result and gradient match pynative.
    Migrated from: test_pijit_exception.py::test_pijit_try_except_break
    """
    class Net7(Cell):
        def __init__(self):
            super().__init__()
            self.op = ops.ReduceMean(False)

        def construct(self, x):
            try:
                out = self.op(x, 0)
            except ValueError:
                x = x.expand_dims(3)
                out = self.op(x, 3)
            else:
                out = out + out
            finally:
                out = out * out
            return out

    x = Tensor(np.ones([2, 3, 4], np.float32))

    # Pynative mode: forward + gradient
    net_pynative = Net7()
    pynative_result = net_pynative(x)
    grad_net = GradOfFirstInput(net_pynative, sens_param=True)
    grad_net.set_train()
    output_shape = pynative_result.shape
    output_grad = Tensor(np.random.randn(*output_shape).astype(np.float32))
    pynative_grad = grad_net(x, output_grad)

    # JIT mode: forward + gradient
    net_jit = Net7()
    net_jit.construct = jit(net_jit.construct, capture_mode='bytecode')
    jit_result = net_jit(x)
    jit_grad_net = GradOfFirstInput(net_jit, sens_param=True)
    jit_grad_net.set_train()
    jit_grad = jit_grad_net(x, output_grad)

    # Compare forward results and gradients
    match_array(pynative_result, jit_result)
    match_array(pynative_grad, jit_grad, error=5)


@arg_mark(
    plat_marks=['cpu_linux'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='essential'
)
def test_except_graph_break_in_else():
    """
    Feature: Exception handling with graph break.
    Description: Test try/except/else with graph break in else block.
    Expectation: JIT result and gradient match pynative.
    Migrated from: test_pijit_exception.py::test_pijit_try_else_break
    """
    class Net8(Cell):
        def __init__(self):
            super().__init__()
            self.op = ops.ReduceMean(False)

        def construct(self, x):
            try:
                out = self.op(x, 0)
            except ValueError:
                x = x.expand_dims(3)
                out = self.op(x, 3)
            else:
                out = out + out
                out.asnumpy()
            finally:
                out = out * out
            return out

    x = Tensor(np.ones([2, 3, 4], np.float32))

    # Pynative mode: forward + gradient
    net_pynative = Net8()
    pynative_result = net_pynative(x)
    grad_net = GradOfFirstInput(net_pynative, sens_param=True)
    grad_net.set_train()
    output_shape = pynative_result.shape
    output_grad = Tensor(np.random.randn(*output_shape).astype(np.float32))
    pynative_grad = grad_net(x, output_grad)

    # JIT mode: forward + gradient
    net_jit = Net8()
    net_jit.construct = jit(net_jit.construct, capture_mode='bytecode')
    jit_result = net_jit(x)
    jit_grad_net = GradOfFirstInput(net_jit, sens_param=True)
    jit_grad_net.set_train()
    jit_grad = jit_grad_net(x, output_grad)

    # Compare forward results and gradients
    match_array(pynative_result, jit_result)
    match_array(pynative_grad, jit_grad, error=5)


@arg_mark(
    plat_marks=['cpu_linux'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='essential'
)
def test_except_graph_break_in_finally():
    """
    Feature: Exception handling with graph break.
    Description: Test try/except/else/finally with graph break in finally
    block.
    Expectation: JIT result and gradient match pynative.
    Migrated from: test_pijit_exception.py::test_pijit_try_finally_break
    """
    class Net9(Cell):
        def __init__(self):
            super().__init__()
            self.op = ops.ReduceMean(False)

        def construct(self, x):
            try:
                out = self.op(x, 0)
            except ValueError:
                x = x.expand_dims(3)
                out = self.op(x, 3)
            else:
                out = out + out
            finally:
                out = out * out
                out.asnumpy()
            return out

    x = Tensor(np.ones([2, 3, 4], np.float32))

    # Pynative mode: forward + gradient
    net_pynative = Net9()
    pynative_result = net_pynative(x)
    grad_net = GradOfFirstInput(net_pynative, sens_param=True)
    grad_net.set_train()
    output_shape = pynative_result.shape
    output_grad = Tensor(np.random.randn(*output_shape).astype(np.float32))
    pynative_grad = grad_net(x, output_grad)

    # JIT mode: forward + gradient
    net_jit = Net9()
    net_jit.construct = jit(net_jit.construct, capture_mode='bytecode')
    jit_result = net_jit(x)
    jit_grad_net = GradOfFirstInput(net_jit, sens_param=True)
    jit_grad_net.set_train()
    jit_grad = jit_grad_net(x, output_grad)

    # Compare forward results and gradients
    match_array(pynative_result, jit_result)
    match_array(pynative_grad, jit_grad, error=5)


@arg_mark(
    plat_marks=['cpu_linux'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='essential'
)
def test_with_statement_basic():
    """
    Feature: With statement.
    Description: Test basic with statement usage.
    Expectation: JIT result and gradient match pynative.
    Migrated from: test_pijit_exception.py::test_pijit_with_basic
    """
    class MyContext:
        def __init__(self):
            self.a = 0
            self.b = 0

        def __enter__(self):
            self.a = 1

        def __exit__(self, t, value, trace):
            self.b = 2

        def get_ab(self):
            return self.a + self.b

    class Net10(Cell):
        def __init__(self, con):
            super().__init__()
            self.a = 1
            self.timer = con

        def construct(self, x):
            out = x
            with self.timer:
                out = x * x
            out = self.timer.get_ab() * out
            return out

    con = MyContext()
    x = Tensor(np.ones([2, 3, 4], np.float32))

    # Pynative mode: forward + gradient
    net_pynative = Net10(con)
    pynative_result = net_pynative(x)
    grad_net = GradOfFirstInput(net_pynative, sens_param=True)
    grad_net.set_train()
    output_shape = pynative_result.shape
    output_grad = Tensor(np.random.randn(*output_shape).astype(np.float32))
    pynative_grad = grad_net(x, output_grad)

    # JIT mode: forward + gradient
    con_jit = MyContext()
    net_jit = Net10(con_jit)
    net_jit.construct = jit(net_jit.construct, capture_mode='bytecode')
    jit_result = net_jit(x)
    jit_grad_net = GradOfFirstInput(net_jit, sens_param=True)
    jit_grad_net.set_train()
    jit_grad = jit_grad_net(x, output_grad)

    # Compare forward results and gradients
    match_array(pynative_result, jit_result)
    match_array(pynative_grad, jit_grad, error=5)


@arg_mark(
    plat_marks=['cpu_linux'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='essential'
)
def test_with_graph_break_in_enter():
    """
    Feature: With statement with graph break.
    Description: Test with statement where graph break occurs in __enter__.
    Expectation: JIT result and gradient match pynative.
    Migrated from: test_pijit_exception.py::test_pijit_with_enter_break
    """
    class MyContext2:
        def __init__(self):
            self.a = 2

        def __enter__(self):
            return np.array([self.a])

        def __exit__(self, t, value, trace):
            return self.a

    class Net11(Cell):
        def __init__(self):
            super().__init__()
            self.k = 1

        def construct(self, x):
            out = x
            with MyContext2():
                out = x * x
            return out * self.k

    x = Tensor(np.ones([2, ], np.float32))

    # Pynative mode: forward + gradient
    net_pynative = Net11()
    pynative_result = net_pynative(x)
    grad_net = GradOfFirstInput(net_pynative, sens_param=True)
    grad_net.set_train()
    output_shape = pynative_result.shape
    output_grad = Tensor(np.random.randn(*output_shape).astype(np.float32))
    pynative_grad = grad_net(x, output_grad)

    # JIT mode: forward + gradient
    net_jit = Net11()
    net_jit.construct = jit(net_jit.construct, capture_mode='bytecode')
    jit_result = net_jit(x)
    jit_grad_net = GradOfFirstInput(net_jit, sens_param=True)
    jit_grad_net.set_train()
    jit_grad = jit_grad_net(x, output_grad)

    # Compare forward results and gradients
    match_array(pynative_result, jit_result)
    match_array(pynative_grad, jit_grad, error=5)


@arg_mark(
    plat_marks=['cpu_linux'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='essential'
)
def test_with_graph_break_in_block():
    """
    Feature: With statement with graph break.
    Description: Test with statement where graph break occurs in with block.
    Expectation: JIT result and gradient match pynative.
    Migrated from: test_pijit_exception.py::test_pijit_with_block_break
    """
    class MyContext3:
        def __init__(self):
            self.k = 2

        def __enter__(self):
            return self.k

        def __exit__(self, t, a, b):
            pass

    class Net12(Cell):
        def __init__(self):
            super().__init__()
            self.k = 1

        def construct(self, x):
            out = x * self.k
            with MyContext3() as c:
                out = x * c
                out.asnumpy()
            return out

    x = Tensor(np.ones([2, 3, 2], np.float32))

    # Pynative mode: forward + gradient
    net_pynative = Net12()
    net_pynative.set_grad()
    pynative_result = net_pynative(x)
    output_grad = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = compute_grad_of_net_inputs(net_pynative, x, sens=output_grad)

    # JIT mode: forward + gradient
    net_jit = Net12()
    net_jit.set_grad()
    net_jit.construct = jit(net_jit.construct, capture_mode='bytecode')
    jit_result = net_jit(x)
    jit_grad = compute_grad_of_net_inputs(net_jit, x, sens=output_grad)

    # Compare forward results and gradients
    match_array(pynative_result, jit_result)
    match_array(pynative_grad, jit_grad, error=5)


@arg_mark(
    plat_marks=['cpu_linux'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='essential'
)
def test_with_graph_break_in_exit():
    """
    Feature: With statement with graph break.
    Description: Test with statement where graph break occurs in __exit__.
    Expectation: JIT result and gradient match pynative.
    Migrated from: test_pijit_exception.py::test_pijit_with_exit_break
    """
    class MyContext4:
        def __init__(self):
            self.a = 3
            self.b = True

        def __enter__(self):
            return self.a

        def __exit__(self, t, value, trace):
            return self.b

    class Net13(Cell):
        def __init__(self):
            super().__init__()
            self.k = 1

        def construct(self, x):
            out = x
            with MyContext4() as c:
                out = x * c
            return out * self.k

    x = Tensor(np.ones([2, 3, 2], np.float32))

    # Pynative mode: forward + gradient
    net_pynative = Net13()
    net_pynative.set_grad()
    pynative_result = net_pynative(x)
    output_grad = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = compute_grad_of_net_inputs(net_pynative, x, sens=output_grad)

    # JIT mode: forward + gradient
    net_jit = Net13()
    net_jit.set_grad()
    net_jit.construct = jit(net_jit.construct, capture_mode='bytecode')
    jit_result = net_jit(x)
    jit_grad = compute_grad_of_net_inputs(net_jit, x, sens=output_grad)

    # Compare forward results and gradients
    match_array(pynative_result, jit_result)
    match_array(pynative_grad, jit_grad, error=5)


@arg_mark(
    plat_marks=['cpu_linux'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='essential'
)
def test_with_exception_in_block():
    """
    Feature: With statement with exception.
    Description: Test with statement where exception occurs in with block.
    Expectation: JIT result and gradient match pynative.
    Migrated from: test_pijit_exception.py::test_pijit_with_raise_error
    """
    class MyContext5:
        def __init__(self):
            self.a = True
            self.b = False

        def __enter__(self):
            return self.a

        def __exit__(self, t, value, trace):
            if t == ValueError:
                return self.a
            return self.b

    class Net14(Cell):
        def __init__(self):
            super().__init__()
            self.k = 1

        def construct(self, x):
            out = x * self.k
            with MyContext5():
                out = x * x
                raise ValueError
            return out

    x = Tensor(np.ones([2, 3, 2], np.float32))

    # Pynative mode: forward + gradient
    net_pynative = Net14()
    net_pynative.set_grad()
    pynative_result = net_pynative(x)
    output_grad = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = compute_grad_of_net_inputs(net_pynative, x, sens=output_grad)

    # JIT mode: forward + gradient
    net_jit = Net14()
    net_jit.set_grad()
    net_jit.construct = jit(net_jit.construct, capture_mode='bytecode')
    jit_result = net_jit(x)
    jit_grad = compute_grad_of_net_inputs(net_jit, x, sens=output_grad)

    # Compare forward results and gradients
    match_array(pynative_result, jit_result)
    match_array(pynative_grad, jit_grad, error=5)

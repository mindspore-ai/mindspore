from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore import jit
from mindspore._c_expression import get_code_extra
import pytest
import sys
from tests.mark_utils import arg_mark

SYS_VER = (sys.version_info.major, sys.version_info.minor)
if SYS_VER != (3, 7) and SYS_VER != (3, 9):
    pytest.skip(reason="not implement for python" + str(SYS_VER), allow_module_level=True)

def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

GEN = fibonacci()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_exception_case_1():
    """
    Feature: Test exception syntax
    Description: Test exception syntax in normal process
    Expectation: No exception.
    """

    def func(val):
        ad = P.Add()
        try:
            val = val + 1
            val = ad(Tensor([val]), Tensor([val]))
        except ValueError:
            next(GEN)
        finally:
            val = ad(val, val)
        return val

    expected = func(5)
    res = jit(fn=func, mode="PIJit")(5)
    jcr = get_code_extra(func)
    assert jcr["code"]["call_count_"] > 0
    assert jcr["compile_count_"] == 1
    assert jcr["break_count_"] == 0
    assert expected == res


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_exception_case_2():
    """
    Feature: Test exception syntax
    Description: Test exception syntax with else block in normal process
    Expectation: No exception.
    """

    def func(val):
        ad = P.Add()
        try:
            val = val + 1
            val = ad(Tensor([val]), Tensor([val]))
        except ValueError:
            next(GEN)
        else:
            val = ad(val, val)
        finally:
            val = ad(val, val)
        return val

    expected = func(5)
    res = jit(fn=func, mode="PIJit")(5)
    jcr = get_code_extra(func)
    assert jcr["code"]["call_count_"] > 0
    assert jcr["compile_count_"] == 1
    assert jcr["break_count_"] == 0
    assert expected == res


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_exception_case_3():
    """
    Feature: Test exception syntax
    Description: Test exception syntax with graph_break point in try block
    Expectation: No exception.
    """

    def func(val):
        ad = P.Add()
        try:
            val = val + 1
            next(GEN) # break
        except ValueError:
            next(GEN)
        else:
            val = ad(Tensor([val]), Tensor([val]))
        finally:
            val = ad(val, val)
        return val

    expected = func(5)
    res = jit(fn=func, mode="PIJit")(5)
    jcr = get_code_extra(func)
    assert jcr["code"]["call_count_"] > 0
    assert jcr["compile_count_"] == 1
    assert jcr["break_count_"] == 1
    assert expected == res


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_exception_case_4():
    """
    Feature: Test exception syntax
    Description: Test exception syntax with graph_break point in else block
    Expectation: No exception.
    """

    def func(val):
        ad = P.Add()
        try:
            val = val + 1
        except ValueError:
            next(GEN)
        else:
            val = ad(Tensor([val]), Tensor([val]))
            next(GEN) # break
        finally:
            val = ad(val, val)
        return val

    expected = func(5)
    res = jit(fn=func, mode="PIJit")(5)
    jcr = get_code_extra(func)
    assert jcr["code"]["call_count_"] > 0
    assert jcr["compile_count_"] == 1
    assert jcr["break_count_"] == 1
    assert expected == res


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_exception_case_5():
    """
    Feature: Test exception syntax
    Description: Test exception syntax with graph_break point in finally block
    Expectation: No exception.
    """

    def func(val):
        ad = P.Add()
        try:
            val = val + 1
        except ValueError:
            next(GEN)
        else:
            val = ad(Tensor([val]), Tensor([val]))
        finally:
            val = ad(val, val)
            next(GEN) # break
        return val

    expected = func(5)
    res = jit(fn=func, mode="PIJit")(5)
    jcr = get_code_extra(func)
    assert jcr["code"]["call_count_"] > 0
    assert jcr["compile_count_"] == 1
    assert jcr["break_count_"] == 1
    assert expected == res


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_exception_case_6():
    """
    Feature: Test exception syntax
    Description: Nested exception syntax
    Expectation: No exception.
    """

    def func(val):
        ad = P.Add()
        try:
            val = val + 1
            try:
                val = val + 1
                val = ad(Tensor([val]), Tensor([val]))
            except ValueError:
                next(GEN)
            finally:
                val = ad(val, val)
        except ValueError:
            next(GEN)
        else:
            val = ad(val, val)
        finally:
            val = ad(val, val)
        return val

    expected = func(5)
    res = jit(fn=func, mode="PIJit")(5)
    jcr = get_code_extra(func)
    assert jcr["code"]["call_count_"] > 0
    assert jcr["compile_count_"] == 1
    assert jcr["break_count_"] == 0
    assert expected == res


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_exception_case_7():
    """
    Feature: Test exception syntax
    Description: Nested exception syntax with a graph_break point in the nested try block
    Expectation: No exception.
    """

    def func(val):
        ad = P.Add()
        try:
            val = val + 1
            try:
                val = val + 1
                next(GEN) # break
            except ValueError:
                next(GEN)
            else:
                val = ad(Tensor([val]), Tensor([val]))
            finally:
                val = ad(val, val)
        except ValueError:
            next(GEN)
        else:
            val = ad(val, val)
        finally:
            val = ad(val, val)
        return val

    expected = func(5)
    res = jit(fn=func, mode="PIJit")(5)
    jcr = get_code_extra(func)
    assert jcr["code"]["call_count_"] > 0
    assert jcr["compile_count_"] == 1
    assert jcr["break_count_"] == 1
    assert expected == res


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_exception_case_8():
    """
    Feature: Test exception syntax
    Description: Nested exception syntax with a graph_break point
    Expectation: No exception.
    """

    def func(val):
        ad = P.Add()
        try:
            val = val + 1
            try:
                val = val + 1
                next(GEN) # break
            except ValueError:
                next(GEN)
            else:
                val = ad(Tensor([val]), Tensor([val]))
            finally:
                val = ad(val, val)
                next(GEN) # break
        except ValueError:
            next(GEN)
        else:
            val = ad(val, val)
            next(GEN) # break
        finally:
            val = ad(val, val)
        return val

    expected = func(5)
    res = jit(fn=func, mode="PIJit")(5)
    jcr = get_code_extra(func)
    assert jcr["code"]["call_count_"] > 0
    assert jcr["compile_count_"] == 1
    assert jcr["break_count_"] == 1
    assert expected == res


class TestWithContext:
    test_value = 0

    def __init__(self, val):
        self.val = val

    def __enter__(self):
        test_value = self.val + 2
        test_value = test_value + 1
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        test_value = self.val + 3
        test_value = self.val - 3
        test_value = test_value - 1


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_exception_case_9():
    """
    Feature: Test exception syntax
    Description: With syntax nested exception syntax
    Expectation: No exception.
    """

    def func(val):
        ad = P.Add()
        with TestWithContext(val):
            val = val + 2
            try:
                val = val + 1
                val = ad(Tensor([val]), Tensor([val]))
            except ValueError:
                next(GEN)
            else:
                val = ad(val, val)
            finally:
                val = ad(val, val)
        return val

    expected = func(5)
    res = jit(fn=func, mode="PIJit")(5)
    jcr = get_code_extra(func)
    assert jcr["code"]["call_count_"] > 0
    assert jcr["compile_count_"] == 1
    assert expected == res


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_exception_case_10():
    """
    Feature: Test exception syntax
    Description: With syntax nested exception syntax has graph_break point
    Expectation: No exception.
    """

    def func(val):
        ad = P.Add()
        with TestWithContext(val):
            val = val + 2
            try:
                val = val + 1
                val = ad(Tensor([val]), Tensor([val]))
                next(GEN) # break
            except ValueError:
                next(GEN)
            else:
                val = ad(val, val)
            finally:
                val = ad(val, val)
        return val

    expected = func(5)
    res = jit(fn=func, mode="PIJit")(5)
    jcr = get_code_extra(func)
    assert jcr["code"]["call_count_"] > 0
    assert jcr["compile_count_"] == 1
    assert jcr["break_count_"] == 1
    assert expected == res


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_exception_case_11():
    """
    Feature: Test exception syntax
    Description: Exception syntax nested with syntax
    Expectation: No exception.
    """

    def func(val):
        ad = P.Add()
        try:
            val = val + 1
            with TestWithContext(val):
                val = val + 2
                val = ad(Tensor([val]), Tensor([val]))
        except ValueError:
            next(GEN)
        else:
            val = ad(val, val)
        finally:
            val = ad(val, val)
        return val

    expected = func(5)
    res = jit(fn=func, mode="PIJit")(5)
    jcr = get_code_extra(func)
    assert jcr["code"]["call_count_"] > 0
    assert jcr["compile_count_"] == 1
    assert jcr["break_count_"] == 0
    assert expected == res


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_exception_case_12():
    """
    Feature: Test exception syntax
    Description: Exception syntax nested with syntax has graph_break point
    Expectation: No exception.
    """

    def func(val):
        ad = P.Add()
        try:
            val = val + 1
            with TestWithContext(val):
                val = val + 2
                val = ad(Tensor([val]), Tensor([val]))
                next(GEN) # break
        except ValueError:
            next(GEN)
        else:
            val = ad(val, val)
        finally:
            val = ad(val, val)
        return val

    expected = func(5)
    res = jit(fn=func, mode="PIJit")(5)
    jcr = get_code_extra(func)
    assert jcr["code"]["call_count_"] > 0
    assert jcr["compile_count_"] == 1
    assert jcr["break_count_"] == 1
    assert expected == res


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_exception_case_13():
    """
    Feature: Test exception syntax
    Description: With syntax nested exception syntax
    Expectation: No exception.
    """

    def func(val):
        ad = P.Add()
        with TestWithContext(val):
            val = val + 2
            try:
                val = val + 1
                try:
                    val = val + 1
                    val = ad(Tensor([val]), Tensor([val]))
                except ValueError:
                    next(GEN)
                finally:
                    val = ad(val, val)
            except ValueError:
                next(GEN)
            else:
                val = ad(val, val)
            finally:
                val = ad(val, val)
        return val

    expected = func(5)
    res = jit(fn=func, mode="PIJit")(5)
    jcr = get_code_extra(func)
    assert jcr["code"]["call_count_"] > 0
    assert jcr["compile_count_"] == 1
    assert expected == res


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_exception_case_14():
    """
    Feature: Test exception syntax
    Description: Exception syntax nested with syntax
    Expectation: No exception.
    """

    def func(val):
        ad = P.Add()
        try:
            val = val + 1
            try:
                val = val + 1
                with TestWithContext(val):
                    val = val + 2
                    val = ad(Tensor([val]), Tensor([val]))
            except ValueError:
                next(GEN)
            else:
                val = ad(val, val)
            finally:
                val = ad(val, val)
        except ValueError:
            next(GEN)
        else:
            val = ad(val, val)
        finally:
            val = ad(val, val)
        return val

    expected = func(5)
    res = jit(fn=func, mode="PIJit")(5)
    jcr = get_code_extra(func)
    assert jcr["code"]["call_count_"] > 0
    assert jcr["compile_count_"] == 1
    assert jcr["break_count_"] == 0
    assert expected == res


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_exception_case_15():
    """
    Feature: Test exception syntax
    Description: Exception syntax nested with syntax has graph_break points
    Expectation: No exception.
    """

    def func(val):
        ad = P.Add()
        try:
            val = val + 1
            try:
                val = val + 1
                with TestWithContext(val):
                    val = val + 2
                    val = ad(Tensor([val]), Tensor([val]))
                next(GEN) # break
            except ValueError:
                next(GEN)
            else:
                val = ad(val, val)
            finally:
                val = ad(val, val)
        except ValueError:
            next(GEN)
        else:
            val = ad(val, val)
            next(GEN) # break
        finally:
            val = ad(val, val)
        return val

    expected = func(5)
    res = jit(fn=func, mode="PIJit")(5)
    jcr = get_code_extra(func)
    assert jcr["code"]["call_count_"] > 0
    assert jcr["compile_count_"] == 1
    assert jcr["break_count_"] == 1
    assert expected == res

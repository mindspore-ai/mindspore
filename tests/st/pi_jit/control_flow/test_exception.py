from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore import jit
from mindspore._c_expression import get_code_extra
import pytest


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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
            print(1)
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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
            print(1)
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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
            print("break!")
        except ValueError:
            print(1)
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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
            print(1)
        else:
            val = ad(Tensor([val]), Tensor([val]))
            print("break!")
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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
            print(1)
        else:
            val = ad(Tensor([val]), Tensor([val]))
        finally:
            val = ad(val, val)
            print("break!")
        return val

    expected = func(5)
    res = jit(fn=func, mode="PIJit")(5)
    jcr = get_code_extra(func)
    assert jcr["code"]["call_count_"] > 0
    assert jcr["compile_count_"] == 1
    assert jcr["break_count_"] == 1
    assert expected == res


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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
                print(1)
            finally:
                val = ad(val, val)
        except ValueError:
            print(1)
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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
                print("break!")
            except ValueError:
                print(1)
            else:
                val = ad(Tensor([val]), Tensor([val]))
            finally:
                val = ad(val, val)
        except ValueError:
            print(1)
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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
                print("break!")
            except ValueError:
                print(1)
            else:
                val = ad(Tensor([val]), Tensor([val]))
            finally:
                val = ad(val, val)
                print("break!")
        except ValueError:
            print(1)
        else:
            val = ad(val, val)
            print("break!")
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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
                print(1)
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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
                print("break!")
            except ValueError:
                print(1)
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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
            print(1)
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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
                print("break!")
        except ValueError:
            print(1)
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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
                    print(1)
                finally:
                    val = ad(val, val)
            except ValueError:
                print(1)
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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
                print(1)
            else:
                val = ad(val, val)
            finally:
                val = ad(val, val)
        except ValueError:
            print(1)
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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
                print("break!")
            except ValueError:
                print(1)
            else:
                val = ad(val, val)
            finally:
                val = ad(val, val)
        except ValueError:
            print(1)
        else:
            val = ad(val, val)
            print("break!")
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

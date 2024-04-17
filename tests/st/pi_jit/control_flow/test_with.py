from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore import jit
from mindspore._c_expression import get_code_extra
import pytest
import dis


class TestWithContext:
    def __init__(self, val):
        self.val = val

    def __enter__(self):
        test_value = self.val + 2
        test_value += 1
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        test_value = self.val + 3
        test_value = self.val - 3
        test_value += 1

class TestWithContext_1:
    def __init__(self, val):
        self.val = val

    def __enter__(self):
        test_value = self.val + 2
        test_value += 1
        print("1")
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        test_value = self.val + 3
        test_value = self.val - 3
        test_value += 1

class TestWithContext_2:
    def __init__(self, val):
        self.val = val

    def __enter__(self):
        test_value = self.val + 2
        test_value += 1
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        test_value = self.val + 3
        print("2")
        test_value = self.val - 3
        test_value += 1

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_with_case_1():
    """
    Feature: Test with syntax
    Description: Test with syntax in normal process
    Expectation: No exception.
    """
    def func(val, add):
        ad = P.Add()
        with TestWithContext(val):
            val = val + add
            ad(Tensor([val]), Tensor([val]))
        add = add + 1
        val = val + 3
        return val + add
    test_value = 0
    expected = func(test_value, 5)
    res = jit(fn=func, mode="PIJit")(test_value, 5)
    jcr = get_code_extra(func)
    assert jcr["code"]["call_count_"] > 0
    assert expected == res

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_with_case_2():
    """
    Feature: Test with syntax
    Description: Test with syntax with break graph
    Expectation: No exception.
    """
    def func(val, add):
        with TestWithContext(val):
            val = val + add
            print("1")
        add = add + 1
        val = val + 3
        return val + add
    test_value = 0
    expected = func(test_value, 5)
    res = jit(fn=func, mode="PIJit")(test_value, 5)
    jcr = get_code_extra(func)
    new_code = jcr["code"]["compiled_code_"]
    flag = False
    for i in dis.get_instructions(new_code):
        if i.opname == "SETUP_WITH":
            flag = True
    assert flag
    assert expected == res

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_with_case_3():
    """
    Feature: Test with syntax
    Description: graph_break in function '__enter__'
    Expectation: no exception.
    """
    def func(val, add):
        with TestWithContext_1(val):
            val = val + add
        add = add + 1
        val = val + 3
        return val + add
    test_value = 0
    expected = func(test_value, 5)
    res = jit(fn=func, mode="PIJit")(test_value, 5)
    jcr = get_code_extra(func)
    assert jcr["code"]["call_count_"] > 0
    assert expected == res

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_with_case_4():
    """
    Feature: Test with syntax
    Description: graph_break in function 'exit'
    Expectation: no exception.
    """
    def func(val, add):
        with TestWithContext_2(val):
            val = val + add
        add = add + 1
        val = val + 3
        return val + add
    test_value = 0
    expected = func(test_value, 5)
    res = jit(fn=func, mode="PIJit")(test_value, 5)
    jcr = get_code_extra(func)
    assert jcr["code"]["call_count_"] > 0
    assert expected == res

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_with_case_5():
    """
    Feature: Test with syntax
    Description: Nested with syntax in normal process
    Expectation: no exception.
    """
    def func(val, add):
        with TestWithContext(val):
            val = val + add
            with TestWithContext(val):
                ad = P.Add()
                val = val + add
                ad(Tensor([val]), Tensor([val]))
        add = add + 1
        val = val + 3
        return val + add
    test_value = 0
    expected = func(test_value, 5)
    res = jit(fn=func, mode="PIJit")(test_value, 5)
    jcr = get_code_extra(func)
    assert jcr["code"]["call_count_"] > 0
    assert expected == res

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_with_case_6():
    """
    Feature: Test with syntax
    Description: Nested with syntax in normal process
    Expectation: no exception.
    """
    def func(val, add):
        with TestWithContext(val):
            val = val + add
            with TestWithContext(val):
                val = val + add
                with TestWithContext(val):
                    ad = P.Add()
                    val = val + add
                    ad(Tensor([val]), Tensor([val]))
                add = add + 1
                val = val + 3
        add = add + 1
        val = val + 3
        return val + add
    test_value = 0
    expected = func(test_value, 5)
    res = jit(fn=func, mode="PIJit")(test_value, 5)
    jcr = get_code_extra(func)
    assert jcr["code"]["call_count_"] > 0
    assert expected == res

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_with_case_7():
    """
    Feature: Test with syntax
    Description: parallel with syntax in normal process
    Expectation: no exception.
    """
    def func(val, add):
        ad = P.Add()
        with TestWithContext(val):
            val = val + add
        add = add + 1
        val = val + 3
        with TestWithContext(add):
            val = val + add
            ad(Tensor([val]), Tensor([val]))
        add = add + 1
        val = val + 3
        return val + add
    test_value = 0
    expected = func(test_value, 5)
    res = jit(fn=func, mode="PIJit")(test_value, 5)
    jcr = get_code_extra(func)
    assert jcr["code"]["call_count_"] > 0
    assert expected == res

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_with_case_8():
    """
    Feature: Test with syntax
    Description: parallel and Nested with syntax in normal process
    Expectation: no exception.
    """
    def func(val, add):
        with TestWithContext(val):
            val = val + add
            with TestWithContext(val):
                ad = P.Add()
                val = val + add
                ad(Tensor([val]), Tensor([val]))
        test_val = 3
        with TestWithContext(test_val):
            dv = P.Div()
            val = val + add
            dv(Tensor([test_val]), Tensor([test_val]))
        return val + test_val + add
    test_value = 0
    expected = func(test_value, 5)
    res = jit(fn=func, mode="PIJit")(test_value, 5)
    jcr = get_code_extra(func)
    assert jcr["code"]["call_count_"] > 0
    assert expected == res

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_with_case_9():
    """
    Feature: Test with syntax
    Description: parallel and Nested with syntax has graph_break
    Expectation: no exception.
    """
    def func(val, add):
        with TestWithContext(val):
            val = val + add
            with TestWithContext(val):
                val = val + add
                print("1")
        test_val = 3
        with TestWithContext(test_val):
            dv = P.Div()
            val = val + add
            dv(Tensor([test_val]), Tensor([test_val]))
        return val + test_val + add
    test_value = 0
    expected = func(test_value, 5)
    res = jit(fn=func, mode="PIJit")(test_value, 5)
    jcr = get_code_extra(func)
    new_code = jcr["code"]["compiled_code_"]
    flag = False
    for i in dis.get_instructions(new_code):
        if i.opname == "SETUP_WITH":
            flag = True
    assert flag
    assert expected == res

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_with_case_10():
    """
    Feature: Test with syntax
    Description: parallel and Nested with syntax has graph_break
    Expectation: no exception.
    """
    def func(val, add):
        with TestWithContext(val):
            val = val + add
            with TestWithContext(val):
                ad = P.Add()
                val = val + add
                ad(Tensor([val]), Tensor([val]))
        test_val = 3
        with TestWithContext(test_val):
            val = val + add
            print("2")
        return val + test_val + add
    test_value = 0
    expected = func(test_value, 5)
    res = jit(fn=func, mode="PIJit")(test_value, 5)
    jcr = get_code_extra(func)
    new_code = jcr["code"]["compiled_code_"]
    flag = False
    for i in dis.get_instructions(new_code):
        if i.opname == "SETUP_WITH":
            flag = True
    assert flag
    assert expected == res

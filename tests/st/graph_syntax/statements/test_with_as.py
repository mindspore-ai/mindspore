# Copyright 2022 Huawei Technologies Co., Ltd
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
""" test graph with as statement. """
import pytest
import mindspore.nn as nn
from mindspore import Tensor, context, jit_class, Parameter

context.set_context(mode=context.GRAPH_MODE)


@jit_class
class Sample1:
    def __init__(self):
        super(Sample1, self).__init__()
        self.num = Tensor([2])

    def __enter__(self):
        return self.num * 2

    def __exit__(self, exc_type, exc_value, traceback):
        print("type:", exc_type)
        print("value:", exc_value)
        print("trace:", traceback)
        return self.num * 4


@jit_class
class Sample2:
    def __init__(self):
        super(Sample2, self).__init__()
        self.num = Tensor([1])

    def __enter__(self):
        return self.num * 3

    def __exit__(self, exc_type, exc_value, traceback):
        print("type:", exc_type)
        print("value:", exc_value)
        print("trace:", traceback)
        return self.num * 5


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_with_basic():
    """
    Feature: Support with as statement.
    Description: Support with as statement.
    Expectation: No exception.
    """
    class TestNet(nn.Cell):
        def construct(self):
            res = 1
            obj = Sample1()
            with obj as sample:
                res += sample
            return res, obj.num

    test_net = TestNet()
    out1, out2 = test_net()
    print("out1:", out1)
    print("out2:", out2)
    assert out1 == 5
    assert out2 == 2


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_with_basic_side_effect():
    """
    Feature: Support with as statement.
    Description: Support with as statement.
    Expectation: No exception.
    """
    class TestNet(nn.Cell):
        def __init__(self):
            super(TestNet, self).__init__()
            self.param = Parameter(Tensor([1]), name="param")

        def construct(self):
            res = 1
            obj = Sample1()
            self.param += 3
            with obj as sample:
                res += obj.num + self.param
                self.param -= sample
            self.param *= 2
            res -= self.param
            return res, obj.num - self.param

    test_net = TestNet()
    out1, out2 = test_net()
    print("out1:", out1)
    print("out2:", out2)
    assert out1 == 7
    assert out2 == 2


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_with_basic_side_effect_2():
    """
    Feature: Support with as statement.
    Description: Support with as statement.
    Expectation: No exception.
    """
    @jit_class
    class SampleParam:
        def __init__(self):
            super(SampleParam, self).__init__()
            self.num = Parameter(Tensor([2]), name="param1")

        def __enter__(self):
            self.num *= 2
            return 0

        def __exit__(self, exc_type, exc_value, traceback):
            print("type:", exc_type)
            print("value:", exc_value)
            print("trace:", traceback)
            self.num *= 4
            return self.num

    class TestNet(nn.Cell):
        def __init__(self):
            super(TestNet, self).__init__()
            self.param = Parameter(Tensor([1]), name="param2")

        def construct(self):
            res = 1
            obj = SampleParam()
            self.param += 3
            with obj as sample:
                res += obj.num + self.param
                self.param -= sample
            self.param *= 2
            res -= self.param
            return res, obj.num - self.param

    test_net = TestNet()
    out1, out2 = test_net()
    print("out1:", out1)
    print("out2:", out2)
    assert out1 == 1
    assert out2 == 8


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_with_basic_parameter():
    """
    Feature: Support with as statement.
    Description: Support with as statement.
    Expectation: No exception.
    """
    @jit_class
    class SampleParam:
        def __init__(self):
            super(SampleParam, self).__init__()
            self.num = Parameter(Tensor([2]), name="param")

        def __enter__(self):
            self.num *= 2
            return self.num

        def __exit__(self, exc_type, exc_value, traceback):
            print("type:", exc_type)
            print("value:", exc_value)
            print("trace:", traceback)
            self.num *= 4
            return self.num

    class TestNet(nn.Cell):
        def construct(self):
            res = 1
            obj = SampleParam()
            with obj as sample:
                res += sample
            return res, obj.num

    test_net = TestNet()
    out1, out2 = test_net()
    print("out1:", out1)
    print("out2:", out2)
    assert out1 == 5
    assert out2 == 16


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_with_pass():
    """
    Feature: Support with as statement.
    Description: Support with as statement.
    Expectation: No exception.
    """
    class TestNet(nn.Cell):
        def construct(self):
            res = 2
            obj = Sample1()
            with obj as sample:
                if sample == 4:
                    pass
            res *= obj.num
            return res, obj.num

    test_net = TestNet()
    out1, out2 = test_net()
    print("out1:", out1)
    print("out2:", out2)
    assert out1 == 4
    assert out2 == 2


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_with_if_return():
    """
    Feature: Support with as statement.
    Description: Support with as statement.
    Expectation: No exception.
    """
    class TestNet(nn.Cell):
        def construct(self, x):
            res = 2
            obj = Sample1()
            with obj as s:
                res = x + s
                if s > 0:
                    return res
            res *= 2
            return res

    test_net = TestNet()
    out = test_net(Tensor([1]))
    print("out:", out)
    assert out == 5


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_with_break():
    """
    Feature: Support with as statement.
    Description: Support with as statement.
    Expectation: No exception.
    """
    class TestNet(nn.Cell):
        def construct(self):
            res = 2
            x = [1, 2, 3, 0, 5]
            obj = Sample1()
            with obj as s:
                y = 1
                for i in x:
                    if i == 0:
                        y += i
                        break
                res += s * y
            return res

    test_net = TestNet()
    out = test_net()
    print("out:", out)
    assert out == 6


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_with_continue():
    """
    Feature: Support with as statement.
    Description: Support with as statement.
    Expectation: No exception.
    """
    class TestNet(nn.Cell):
        def construct(self):
            res = 2
            x = [1, 2, 3, 0, 5]
            obj = Sample1()
            with obj as s:
                y = 1
                for i in x:
                    if i == 3:
                        continue
                    y += i
                res += s * y
            return res

    test_net = TestNet()
    out = test_net()
    print("out:", out)
    assert out == 38


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_with_twice():
    """
    Feature: Support with as statement.
    Description: Support with as statement.
    Expectation: No exception.
    """
    class TestNet(nn.Cell):
        def construct(self):
            res = 2
            obj = Sample1()
            with obj as s:
                res += s
            with obj as s:
                res += s * 2
            return res

    test_net = TestNet()
    out = test_net()
    print("out:", out)
    assert out == 14


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_with_nested():
    """
    Feature: Support with as statement.
    Description: Support with as statement.
    Expectation: No exception.
    """
    class TestNet(nn.Cell):
        def construct(self):
            res = 2
            obj = Sample1()
            with obj as s1:
                res += s1
                with obj as s2:
                    res += s2 * 2
            return res

    test_net = TestNet()
    out = test_net()
    print("out:", out)
    assert out == 14


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_with_parallel():
    """
    Feature: Support with as statement.
    Description: Support with as statement.
    Expectation: No exception.
    """
    class TestNet(nn.Cell):
        def construct(self):
            res = 2
            obj1 = Sample1()
            obj2 = Sample2()
            with obj1 as s1, obj2 as s2:
                res += s1 * s2
            return res

    test_net = TestNet()
    out = test_net()
    print("out:", out)
    assert out == 14


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_with_without_as():
    """
    Feature: Support with as statement.
    Description: Support with as statement.
    Expectation: No exception.
    """
    class TestNet(nn.Cell):
        def construct(self):
            res = 10
            obj1 = Sample1()
            with obj1:
                res = obj1.num
            return res

    test_net = TestNet()
    out = test_net()
    print("out:", out)
    assert out == 2


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_with_as_exception():
    """
    Feature: Support with as statement.
    Description: Support with as statement.
    Expectation: No exception.
    """
    @jit_class
    class Sample():
        def __init__(self):
            super(Sample, self).__init__()
            self.num = Tensor([1])

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            print("type:", exc_type)
            print("value:", exc_value)
            print("trace:", traceback)
            return self.do_something(1)

        def do_something(self, x):
            bar = 2 / 0 + x + self.num
            return bar + 10

    class TestNet(nn.Cell):
        def construct(self, x):
            a = 1
            with Sample() as sample:
                a = sample.do_something(a + x)
            return x * a

    with pytest.raises(ValueError) as as_exception:
        x = Tensor([1])
        test_net = TestNet()
        res = test_net(x)
        print("res:", res)
        assert res == 10
    assert "The divisor could not be zero" in str(as_exception.value)

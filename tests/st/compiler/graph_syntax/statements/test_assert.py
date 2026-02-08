# Copyright 2022-2025 Huawei Technologies Co., Ltd
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
""" test_assert """
import pytest
from mindspore import nn, context, Tensor
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_assert1():
    """
    Feature: support assert
    Description: test assert
    Expectation: AssertionError
    """

    class Net(nn.Cell):
        def construct(self):
            x = 1
            assert x == 2
            return x

    net = Net()
    with pytest.raises(AssertionError):
        net()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_assert2():
    """
    Feature: support assert
    Description: test assert
    Expectation: no error
    """

    class Net(nn.Cell):
        def construct(self):
            x = 1
            assert True
            return x

    net = Net()
    out = net()
    assert out == 1


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_assert3():
    """
    Feature: support assert
    Description: test assert
    Expectation: no error
    """

    class Net(nn.Cell):
        def construct(self):
            x = 1
            assert x in [2, 3, 4]
            return x

    net = Net()
    with pytest.raises(AssertionError):
        net()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_assert4():
    """
    Feature: support assert
    Description: test assert
    Expectation: no error
    """

    class Net(nn.Cell):
        def construct(self):
            x = 1
            assert x in [2, 3, 4], "x not in [2, 3, 4]"
            return x

    net = Net()
    with pytest.raises(AssertionError) as excinfo:
        net()
    assert "x not in [2, 3, 4]" in str(excinfo.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_assert5():
    """
    Feature: support assert
    Description: test assert
    Expectation: no error
    """

    class Net(nn.Cell):
        def construct(self):
            x = 1
            assert x in [2, 3, 4], f"%d not in [2, 3, 4]" % x
            return x

    net = Net()
    with pytest.raises(AssertionError) as excinfo:
        net()
    assert "1 not in [2, 3, 4]" in str(excinfo.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_assert6():
    """
    Feature: support assert
    Description: test assert
    Expectation: no error
    """

    class Net(nn.Cell):
        def construct(self):
            x = 1
            assert x in [2, 3, 4], f"{x} not in [2, 3, 4]"
            return x

    net = Net()
    with pytest.raises(AssertionError) as excinfo:
        net()
    assert "1 not in [2, 3, 4]" in str(excinfo.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_assert7():
    """
    Feature: support assert
    Description: test assert
    Expectation: no error
    """

    class Net(nn.Cell):
        def construct(self):
            x = 1
            assert x in [2, 3, 4], "{} not in [2, 3, 4]".format(x)
            return x

    net = Net()
    with pytest.raises(AssertionError) as excinfo:
        net()
    assert "1 not in [2, 3, 4]" in str(excinfo.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parser_fallback_assert_x_dtype_not_same():
    """
    Feature: support assert which type is not same.
    Description: test assert.
    Expectation: no error.
    """
    class Net(nn.Cell):
        def construct(self):
            x = 2.0
            assert x == 2
            return x

    net = Net()
    net()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parser_fallback_assert_isinstance():
    """
    Feature: support assert with isinstance.
    Description: test assert.
    Expectation: no error.
    """
    class Net(nn.Cell):
        def construct(self):
            x = Tensor([1])
            assert isinstance(x, Tensor)
            return x

    net = Net()
    net()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parser_fallback_assert_isinstance_type_diff():
    """
    Feature: support assert with isinstance.
    Description: test assert.
    Expectation: no error.
    """
    class Net(nn.Cell):
        def construct(self):
            x = [1, 2, 3]
            assert isinstance(x, tuple)
            return x

    net = Net()
    with pytest.raises(AssertionError) as exc_info:
        net()
    assert "assert isinstance(x, tuple)" in str(exc_info.value)

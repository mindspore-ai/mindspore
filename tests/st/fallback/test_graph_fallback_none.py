# Copyright 2023 Huawei Technologies Co., Ltd
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
import pytest
import numpy as np

from mindspore import Tensor, jit, context, Parameter
from mindspore.nn import Cell
from mindspore.nn.probability import distribution
import mindspore.nn as nn
import mindspore.common.dtype as mstype
import mindspore as ms

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_none_compare():
    """
    Feature: Support None.
    Description: Support None compare with string, bool, and empty list.
    Expectation: No exception.
    """
    @jit
    def foo():  # pylint: disable=R1711
        a = ''
        b = False
        c = []
        d = 0
        print(a is None)
        print(b is None)
        print(c is None)
        print(d is None)
        return None

    res = foo()
    print("res:", res)
    assert res is None


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_none_is_sequence_input():
    """
    Feature: Support None.
    Description: Support None is the input of sequence.
    Expectation: No exception.
    """
    @jit
    def foo(input_tuple, input_list, input_dict):
        num = 0
        for input_t in input_tuple:
            if input_t is None:
                num = num + 1
        for input_l in input_list:
            if input_l is None:
                num = num + 1
        for key in input_dict:
            if input_dict[key] is None:
                num = num + 1
        return num

    input_x = (None, None, 1, 2, 3, 4)
    input_y = [None, None, None]
    input_z = {"a": None, "b": None, "c": 5, "d": 6, "e": 7, "f": 8}
    res = foo(input_x, input_y, input_z)
    assert res == 7


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_inner_function_has_not_return():
    """
    Feature: Support None.
    Description: Support has no return function.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = 3
        print("x:", x)

    res = foo()
    assert res is None


@pytest.mark.skip(reason="No support assert.")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_inner_function_has_not_return_2():
    """
    Feature: Support None.
    Description: Support None is the output of inner function.
    Expectation: No exception.
    """
    def func1():
        x = 3
        print("x:", x)

    def func2():
        return None  # pylint: disable=R1711

    def func3():
        return

    @jit
    def foo():
        obj1 = func1()  # pylint: disable=E1111
        obj2 = func2()  # pylint: disable=E1128
        obj3 = func3()  # pylint: disable=E1128
        assert obj1 is None
        assert obj2 is None
        assert obj3 is None
        return 0

    res = foo()
    assert res == 0


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_none_is_default_value_of_parameter():
    """
    Feature: Support None.
    Description: Support None is the default value of parameter.
    Expectation: No exception.
    """
    @jit
    def foo(x, y=None):
        if y is not None:
            print("y:", y)
        else:
            print("y is None")
        print("x:", x)
        return y

    x = [1, 2]
    res1 = foo(x)
    assert res1 is None


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_none_is_default_value_of_parameter_2():
    """
    Feature: Support None.
    Description: Support None is the default value of parameter.
    Expectation: No exception.
    """
    @jit
    def foo(x, y=None):
        if y is not None:
            print("y:", y)
        else:
            print("y is None")
        print("x:", x)
        return y

    x = [1, 2]
    y = [3, 4]
    res2 = foo(x, y)
    assert res2 == [3, 4]


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_none_is_slice_in_list():
    """
    Feature: Support None.
    Description: Support None is slice in list.
    Expectation: No exception.
    """
    @jit
    def foo():
        arr1 = np.array([1, 2, 3])
        print(arr1[0:2])
        print(arr1[:, None])
        print(arr1[None, :])
        arr2 = np.array([[[1, 2], [3, 4], [5, 6]],
                         [[1, 2], [3, 4], [5, 6]]])
        print(arr2[0:2, :])
        print(arr2[:, 0])
        print(arr2[0, :])
        print(arr2[None, :])
        print(arr2[:, None, :])
        print(arr2[:, :, None])
        return 0

    res = foo()
    assert res == 0


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_none_assign_print():
    """
    Feature: Support None.
    Description: Support None assign and print.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = None
        print("x:", x)
        print("convert bool:", bool(None))
        return x

    res = foo()
    assert res is None


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_none_is_input():
    """
    Feature: Support None.
    Description: Support None is arg of top graph.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        return x

    input_x = None
    res = foo(input_x)
    assert res is None


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_none_is_condition():
    """
    Feature: Support None.
    Description: Support None is condition.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        if not x:
            print("The input is None!")
        else:
            print("The input is not None!")
        return x

    input_x = None
    res = foo(input_x)
    assert res is None


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_none_is_inner_function_output():
    """
    Feature: Support None.
    Description: Support None is the output of inner function.
    Expectation: No exception.
    """
    def inner_fun1(input_x):
        if input_x == 0:
            return None, input_x
        return None, input_x * 2

    @jit
    def foo(input_x):  # pylint: disable=R1711
        out1 = inner_fun1(input_x)
        if not out1[1]:
            print('The output of inner function is None!')
        else:
            print('The output of inner function is not None!')
        return None

    x = Tensor([1])
    res = foo(x)
    assert res is None


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_none_is_output_of_function_with_side_effect():
    """
    Feature: Support None.
    Description: Support None is the output of_function with side effect.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.param = Parameter(Tensor([1], dtype=mstype.int64), name="param")

        def func(self, y):  # pylint: disable=R1711
            if y == self.param:
                self.param = y * 2
            return None

        def construct(self, x):
            if self.func(x) is None:
                self.param = self.param + 2 * x
            else:
                self.param = self.param + 3 * x
            return self.param

    net = Net()
    input_x = Tensor([1], dtype=mstype.int32)
    res = net(input_x)
    print("res:", res)
    assert res == 4


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_none_is_input_of_dict_return():
    """
    Feature: Support None.
    Description: Support None is input of dict, and the dict is return.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = {'a': 'a', 'b': 'b'}
        y = x.get('a')
        z = dict(y=y, u=9, v=False, w=None)
        return z

    out = foo()
    assert out == {'y': 'a', 'u': 9, 'v': False, 'w': None}


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_none_nested_input_of_dict_return():
    """
    Feature: Support None.
    Description: Support None is input of dict, and the dict is return.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = {'a': 'a', 'b': 'b'}
        y = x.get('a')
        z = dict(y=y, u=9, v=False, w=(None, None), q=[1, (2, None), None])
        return z

    out = foo()
    assert out == {'y': 'a', 'u': 9, 'v': False, 'w': (None, None), 'q': (1, (2, None), None)}


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_none_is_input_of_tuple_return():
    """
    Feature: Support None.
    Description: Support None is input of tuple, and the tuple is return.
    Expectation: No exception.
    """

    @jit
    def foo():
        return 1, "a", None

    out = foo()
    assert out == (1, "a", None)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_none_is_input_of_tuple_return_2():
    """
    Feature: Support None.
    Description: Support None is input of tuple, and the tuple is return.
    Expectation: No exception.
    """
    class BernoulliCrossEntropy(Cell):
        def __init__(self, probs, seed=10, dtype=ms.int32, name='Bernoulli', dist='Bernoulli'):
            super().__init__()
            self.b = distribution.Bernoulli(probs, seed, dtype, name)
            self.dist = dist

        def construct(self, probs1_b, probs1=None):
            if probs1 is None:
                out1 = self.b.cross_entropy(self.dist, probs1_b)
                out2 = self.b.kl_loss(self.dist, probs1_b)
            else:
                out1 = self.b.cross_entropy(self.dist, probs1_b, probs1)
                out2 = self.b.kl_loss(self.dist, probs1_b, probs1)
            out3 = self.b.probs
            return out1, out2, out3

    probs = None
    probs1 = 0.2
    probs1_b = 0.9
    context.set_context(mode=context.GRAPH_MODE)
    net_graph = BernoulliCrossEntropy(probs)
    out_me_graph = net_graph(Tensor(probs1_b), Tensor(probs1))
    print("out_me_graph: ", out_me_graph)
    context.set_context(mode=context.PYNATIVE_MODE)
    net_pynative = BernoulliCrossEntropy(probs)
    out_me_pynative = net_pynative(Tensor(probs1_b), Tensor(probs1))
    print("out_me_pynative: ", out_me_pynative)
    assert out_me_graph == out_me_pynative


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_none_is_return_of_sub_graph_control_flow():
    """
    Feature: Support None.
    Description: Support None is the return of sub_graph in control flow.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def check_value(self, x):  # pylint: disable=R1711
            if x[0][0] < 2:
                print("The input is less 2.")
            return None

        def construct(self, x):
            self.check_value(x)
            return x

    net = Net()
    data = Tensor(np.ones([2, 3]), dtype=ms.float32)
    out = net(data)
    assert (out.asnumpy() == data).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_none_is_return_of_sub_graph_control_flow_raise():
    """
    Feature: Support None.
    Description: Support None is the return of sub_graph in control flow. And Raise node is in sub_graph.
    Expectation: No exception.
    """
    class RaiseNet(nn.Cell):
        def inner_func(self, x):  # pylint: disable=R1711
            if x == 2:
                raise ValueError("The input should not be ", x)
            return None

        def construct(self, x):
            self.inner_func(x)
            return x

    net = RaiseNet()
    res = net(Tensor(1))
    print("res:", res)
    assert res.asnumpy() == 1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_none_is_return_raise():
    """
    Feature: Support None.
    Description: Support None is the return of sub_graph in control flow. And Raise node is in sub_graph.
    Expectation: No exception.
    """
    def check_test(shp):  # pylint: disable=R1711
        if shp[0] > 5:
            raise ValueError('raise value error.')
        return None

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.one = Tensor(1, dtype=ms.float32)

        def construct(self, x):
            shp = x.shape
            check_test(shp)
            return x

    with pytest.raises(ValueError, match="raise value error."):
        np_data = np.random.randint(6, size=(6,))
        data = Tensor(np_data, dtype=ms.float32)
        dyn_tensor = Tensor(shape=[None], dtype=ms.float32)
        net = Net()
        net.set_inputs(dyn_tensor)
        out = net(data)
        assert out == data

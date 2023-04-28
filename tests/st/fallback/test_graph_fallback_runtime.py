# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
""" test graph JIT Fallback runtime feature """
import math
from functools import reduce
import pytest
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore import ops
from mindspore import mutable, jit

ms.set_context(mode=ms.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_getattr_tensor_with_wrong_attr():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support tensor input.
    Expectation: AttributeError.
    """

    @ms.jit
    def foo(x):
        abs_func = getattr(x, "abs2")
        return abs_func()

    with pytest.raises(AttributeError) as err:
        foo(ms.Tensor([-1, -2, -3]))  # Not throw error any more, should move to ST.
    assert "object has no attribute" in str(err.value)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_getattr_list_with_wrong_attr():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support list input.
    Expectation: AttributeError.
    """

    @ms.jit
    def foo(x):
        abs_func = getattr(x, "abs2")
        return abs_func()

    with pytest.raises(AttributeError) as err:
        foo([1, 2, 3, 4])  # Not throw error any more, should move to ST.
    assert "object has no attribute" in str(err.value)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_getattr_tuple_with_wrong_attr():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support tensor input.
    Expectation: AttributeError.
    """

    @ms.jit
    def foo(x):
        abs_func = getattr(x, "shape")
        return abs_func()

    with pytest.raises(AttributeError) as err:
        foo((1, 2, 3, 4))  # Not throw error any more, should move to ST.
    assert "object has no attribute" in str(err.value)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_getattr_dict_with_wrong_attr():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support tensor input.
    Expectation: AttributeError.
    """

    @ms.jit
    def foo(x):
        abs_func = getattr(x, "abs2")
        return abs_func()

    with pytest.raises(AttributeError) as err:
        foo({"1": 1, "2": 2})  # Not throw error any more, should move to ST.
    assert "object has no attribute" in str(err.value)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_pyexecute_with_scalar_input():
    """
    Feature: Fallback runtime.
    Description: The pyexecute node has scalar input.
    Expectation: No error.
    """
    def _check_is_inf_nan(x):
        if math.isinf(x) or math.isnan(x) or np.isinf(x) or np.isnan(x):
            return True
        return False

    class InnerNet(nn.Cell):
        def construct(self, x):
            return _check_is_inf_nan(x.shape[0])

    net = InnerNet()
    data = Tensor(np.random.randint(6, size=(2, 4, 3, 4, 5)), dtype=ms.float32)
    dyn = Tensor(shape=[None, None, None, None, None], dtype=ms.float32)
    net.set_inputs(dyn)
    ret = net(data)
    assert not ret


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_pyexecute_with_scalar_input_2():
    """
    Feature: Fallback runtime.
    Description: The pyexecute node has scalar input.
    Expectation: No error.
    """
    def _check_is_inf_nan(x):
        if math.isinf(x) or math.isnan(x) or np.isinf(x) or np.isnan(x):
            return True
        return False

    class InnerNet(nn.Cell):
        def construct(self, x):
            return _check_is_inf_nan(x)

    net = InnerNet()
    ret = net(math.inf)
    assert ret


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_pyexecute_with_scalar_input_3():
    """
    Feature: Fallback runtime.
    Description: The pyexecute node has scalar input.
    Expectation: No error.
    """

    class InnerNet(nn.Cell):
        def construct(self, x):
            shp = x.shape
            return all(i < 3 for i in shp)

    net = InnerNet()
    data = Tensor(np.random.randint(6, size=(2, 4, 3, 4, 5)), dtype=ms.float32)
    dyn = Tensor(shape=[None, None, None, None, None], dtype=ms.float32)
    net.set_inputs(dyn)
    ret = net(data)
    assert not ret


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_pyexecute_with_scalar_input_4():
    """
    Feature: Fallback runtime.
    Description: The pyexecute node has scalar input.
    Expectation: No error.
    """

    class InnerNet(nn.Cell):
        def construct(self, x):
            shp = x.shape
            return any(i < 3 for i in shp)

    net = InnerNet()
    data = Tensor(np.random.randint(6, size=(2, 4, 3, 4, 5)), dtype=ms.float32)
    dyn = Tensor(shape=[None, None, None, None, None], dtype=ms.float32)
    net.set_inputs(dyn)
    ret = net(data)
    assert ret


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_pyexecute_as_multitype_fg_input():
    """
    Feature: Fallback runtime.
    Description: Pyexecute node can not be used as multitype function graph.
    Expectation: No error.
    """
    class sub_class:
        def __getitem__(self, item):
            pass
        def __setitem__(self, key, target):
            pass


    class InnerNet(nn.Cell):
        def __init__(self, tuple_input):
            super(InnerNet, self).__init__()
            self.data = tuple_input

        def construct(self, start):
            return self.data[start:]

    sub_class_obj = sub_class()
    sub_class_obj[0] = [1, 2, 3, 4, 5]
    net = InnerNet(sub_class_obj)
    assert net(1) is None


def user_mul(x, y):
    return x * y


@ms.jit
def reduce_user_mul(x):
    out = reduce(user_mul, x)
    return out


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_pyexecute_with_func_graph_input():
    """
    Feature: Fallback runtime.
    Description: The pyexecute node has FuncGraph input.
    Expectation: No error.
    """
    x1 = (1, 2, 3)
    x2 = mutable((1, 2, 3), False)
    ret1 = reduce_user_mul(x1)
    ret2 = reduce_user_mul(x2)
    assert ret1 == 6
    assert ret2 == 6


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_anytype():
    """
    Feature: Fallback runtime.
    Description: test ops input is PyExecute out
    Expectation: No error.
    """

    @jit
    def func(x):
        x = x.asnumpy()
        x = ms.Tensor(x)
        x = ops.ReLU()(x)
        return x

    def func_numpy(x):
        return np.maximum(x, 0)

    x_np = np.array([1, -1])
    ms_out = func(ms.Tensor(np.array([1, -1])))
    np_out = func_numpy(x_np)
    assert np.allclose(np_out, ms_out.asnumpy())


class CreateDynTensor(nn.Cell):
    def construct(self, x):
        # @jit.typing: () -> tensor[int32]
        shape_tensor1 = Tensor(ms.mutable(ops.shape(x)), ms.int32)
        output1 = ops.FillV2()(shape_tensor1, Tensor(1, ms.int32))

        shape_tensor2 = Tensor(ms.mutable(ops.shape(x)), ms.int32)  # @jit.typing: () -> tensor[int32]
        output2 = ops.FillV2()(shape_tensor2, Tensor(1, ms.int32))
        return output1 + output2


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_shape_tensor():
    """
    Feature: Fallback runtime.
    Description: Set PyExecute output type by the annotation from comment.
    Expectation: No error.
    """
    net = CreateDynTensor()
    x = Tensor(dtype=ms.int32, input_data=[2, 2])
    out = net(x)
    return out

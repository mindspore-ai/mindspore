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
from mindspore import Tensor, jit, context, nn

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_getattr_asnumpy():
    """
    Feature: JIT Fallback
    Description: Test getattr in fallback runtime
    Expectation: No exception.
    """
    @jit
    def foo():
        x = Tensor(np.array([1, 2, 3, 4])).asnumpy()
        len_func1 = getattr(x, "__len__", Tensor([-1]))
        attr = "__len__"
        len_func2 = getattr(x, attr)
        return len_func1(), len_func2()

    out = foo()
    assert out[0] == out[1] == 4


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_getattr_asnumpy_custom_class():
    """
    Feature: getattr for custom class.
    Description: Support getattr for custom class.
    Expectation: No exception.
    """
    class GetattrClass():
        def __init__(self):
            self.attr1 = Tensor(np.array([1, 2, 3, 4])).asnumpy()
            self.attr2 = 1

    class GetattrClassNet(nn.Cell):
        def __init__(self):
            super(GetattrClassNet, self).__init__()
            self.cls = GetattrClass()

        def construct(self):
            attr = "__len__"
            len_func1 = getattr(self.cls.attr1, attr)
            len_func2 = getattr(self.cls.attr1, "__len__")
            return len_func1(), len_func2()

    net = GetattrClassNet()
    out = net()
    assert out[0] == out[1] == 4


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_getattr_numpy_array_2():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support numpy array input.
    Expectation: TypeError
    """

    @jit
    def foo():
        x = 1
        return getattr(x, "shape", np.array([0, 1, 2, 3, 4]))

    out = foo()
    assert (out == np.array([0, 1, 2, 3, 4])).all()

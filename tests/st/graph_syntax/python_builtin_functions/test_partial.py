# Copyright 2020 Huawei Technologies Co., Ltd
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
""" test partial"""
from functools import partial
import pytest

from mindspore import nn, context

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_partial_key_ward_arg_and_pos_arg_const_multi_assign_x():
    """
    Feature: ALL TO ALL
    Description: test cases for partial_key_ward_arg_and_pos_arg_const_multi_assign_x
    Expectation: the result match given one
    """

    class Net(nn.Cell):
        def show(self, x, y, z):
            return x, y, z

        def construct(self):
            f = partial(self.show, x=1)
            ret = f(1, 2, 3)
            return ret

    class Net2(nn.Cell):
        def __init__(self):
            super(Net2, self).__init__()
            self.show = lambda x, y, z: (x, y, z)

        def construct(self):
            f = partial(self.show, x=1)
            ret = f(1, 2, 3)
            return ret

    for net in [Net(), Net2()]:
        with pytest.raises(TypeError) as ex:
            net()
        assert "got multiple values for argument 'x'" or \
               "Multiply values for specific argument: x" in str(ex.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_partial_key_ward_arg_and_pos_arg_const_multi_assign_y():
    """
    Feature: ALL TO ALL
    Description: test cases for partial_key_ward_arg_and_pos_arg_const_multi_assign_y
    Expectation: the result match given one
    """

    class Net(nn.Cell):
        def show(self, x, y, z):
            return x, y, z

        def construct(self):
            f = partial(self.show, y=2)
            ret = f(1, 2, z=3)
            return ret

    class Net2(nn.Cell):
        def __init__(self):
            super(Net2, self).__init__()
            self.show = lambda x, y, z: (x, y, z)

        def construct(self):
            f = partial(self.show, y=2)
            ret = f(1, 2, z=3)
            return ret

    for net in [Net(), Net2()]:
        with pytest.raises(TypeError) as ex:
            net()
        assert "got multiple values for argument 'z'" or \
               "Multiply values for specific argument: z" in str(ex.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_partial_key_ward_arg_and_pos_arg_const_multi_assign_z():
    """
    Feature: ALL TO ALL
    Description: test cases for partial_key_ward_arg_and_pos_arg_const_multi_assign_z
    Expectation: the result match given one
    """

    class Net(nn.Cell):
        def show(self, x, y, z):
            return x, y, z

        def construct(self):
            f = partial(self.show, z=1)
            ret = f(1, 2, 3)
            return ret

    class Net2(nn.Cell):
        def __init__(self):
            super(Net2, self).__init__()
            self.show = lambda x, y, z: (x, y, z)

        def construct(self):
            f = partial(self.show, z=1)
            ret = f(1, 2, 3)
            return ret

    for net in [Net(), Net2()]:
        with pytest.raises(TypeError) as ex:
            net()
        assert "got multiple values for argument 'z'" or \
               "Multiply values for specific argument: z" in str(ex.value)

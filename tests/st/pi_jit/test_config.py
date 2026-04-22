# Copyright 2024 Huawei Technologies Co., Ltd
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
"""test jit config"""
from mindspore import Tensor, jit
from .share.utils import assert_executed_by_graph_mode
from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import pi_jit_with_config
from mindspore._c_expression import function_id, TensorPy as Tensor_

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_config_disable_pijit():
    """
    Feature: Jit config
    Description: Jit config
    Expectation: The result match
    """
    @pi_jit_with_config(jit_config={'_disable_pijit':lambda args, kwds: args[0] > 1})
    def func(x, y):
        return x + y

    for i in range(10):
        a = Tensor([i])
        func(a, a)

    assert_executed_by_graph_mode(func, call_count=2)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_config_function_id_map():
    """
    Feature: Test function_id
    Description: function_id must be return a function identify instead of method
    Expectation: The result match
    """
    class Test:
        @classmethod
        def __subclasshook__(cls, sub):
            pass
        @staticmethod
        def from_numpy(input_data):
            pass
        def __init__(self):
            pass

    assert function_id(tuple.__getitem__) == function_id(tuple().__getitem__), "check WrapperDescriptor failed"
    assert function_id(list.__getitem__) == function_id([].__getitem__), "check MethodDescriptor failed"
    assert function_id(Tensor_.from_numpy) == function_id(Tensor_(1).from_numpy), "check pybind11 instancemethod failed"
    assert function_id(Tensor_.from_numpy) != function_id(Tensor_.asnumpy), "check pybind11 instancemethod failed"
    assert function_id(Test.__init__) == function_id(Test().__init__) == id(Test.__init__), "check python function id failed"
    assert function_id(Test) == id(Test), "check user defined object id failed"
    assert function_id(Test.__subclasshook__) == id(Test.__subclasshook__.__func__), "check classmethod failed"
    assert function_id(Test.from_numpy) == id(Test().from_numpy), "check staticmethod failed"

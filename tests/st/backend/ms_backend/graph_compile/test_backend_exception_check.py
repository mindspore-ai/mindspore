# Copyright 2025 Huawei Technologies Co., Ltd
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
# ==============================================================================
'''
Check exception for backend compile.
'''
import pytest
import mindspore as ms
from mindspore import Tensor, context
from mindspore.ops import operations as P
from tests.st.backend.ms_backend.common.backend_graph import BackendGraph
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_check_exception_in_pass():
    """
    Feature: Check exception for backend compile.
    Description: Check exception for tensor to tuple after kernel select.
    Expectation: Run success.
    """
    context.set_context(device_target="CPU")
    a = BackendGraph()
    shape = (2, 2)
    para_1 = a.add_parameter(ms.float32, shape)
    tensor_2 = a.add_valuenode(Tensor(2))
    mul = a.add_cnode(a.add_valuenode(P.Mul()), tensor_2, tensor_2)
    a.set_abstract(mul, ms.int64, ())
    reshape = a.add_cnode(a.add_valuenode(P.Reshape()), para_1, mul)
    a.set_abstract(reshape, ms.float32, (-2,))
    a.add_return(reshape)
    a.skip_infer()
    print(a)
    with pytest.raises(RuntimeError) as err:
        a.compile()
    assert "Invalid kernel object type" in str(err.value)

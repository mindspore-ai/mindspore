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
""" test_list_extend """
import pytest
import numpy as np
import mindspore as ms
from tests.mark_utils import arg_mark


@pytest.mark.skip(reason="empty list used as PyExecute input is not supported yet")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_extend_tensor():
    """
    Feature: list extend.
    Description: support list extend.
    Expectation: No exception.
    """
    @ms.jit
    def func():
        x = []
        y = ms.Tensor([[1, 2], [3, 4]])
        x.extend(y)
        return x

    out = func()
    assert np.all(out[0].asnumpy() == ms.Tensor([1, 2]).asnumpy())
    assert np.all(out[1].asnumpy() == ms.Tensor([3, 4]).asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_extend_tensor_2():
    """
    Feature: list extend.
    Description: support list extend.
    Expectation: No exception.
    """
    @ms.jit
    def func():
        x = [1,]
        y = ms.Tensor([[1, 2], [3, 4]])
        x.extend(y)
        return x

    out = func()
    assert isinstance(out, list)
    assert len(out) == 3
    assert np.all(out[0] == 1)
    assert np.all(out[1].asnumpy() == ms.Tensor([1, 2]).asnumpy())
    assert np.all(out[2].asnumpy() == ms.Tensor([3, 4]).asnumpy())

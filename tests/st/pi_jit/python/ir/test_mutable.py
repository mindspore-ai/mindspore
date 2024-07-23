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
"""test mutable"""
import pytest
from mindspore.common import mutable
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore import jit, context
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_mutable_with_scalar():
    """
    Feature: Set Constants mutable.
    Description: Set mutable for scalar.
    Expectation: No Exception.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    mutable(1)
    mutable([Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32), (2,)])
    mutable({'a': Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32), 'b': (2,)})
    @jit(mode="PIJit")
    def net():
        x = mutable(2)
        return x
    net()

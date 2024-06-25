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
from mindspore import context, jit
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_string_operate():
    """
    Feature: Support is.
    Description: Support string operate in fallback runtime.
    Expectation: No exception.
    """
    @jit
    def foo():
        var1 = 'Hello!'
        var2 = "MindSpore"
        out1 = var1[0]
        out2 = var2[4:9]
        out3 = var1 + var2
        out4 = var2 * 2
        out5 = str("H" in var1)
        out6 = "My name is %s!" % var2
        return out1 + "_" + out2 + "_" + out3 + "_" + out4 + "_" + out5 + "_" + out6

    res = foo()
    assert res == "H_Spore_Hello!MindSpore_MindSporeMindSpore_True_My name is MindSpore!"

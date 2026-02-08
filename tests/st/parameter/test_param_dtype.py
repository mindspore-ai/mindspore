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
# ============================================================================
import pytest

import mindspore as ms
from mindspore import context, Parameter
from mindspore.nn.utils import no_init_parameters
from mindspore.common.initializer import initializer
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_parameter_set_dtype(mode):
    """
    Feature: mindspore.parameter.dtype
    Description: Uninitialized parameters use different types from initialization. Verify if type setting is effective.
    Expectation: success
    """
    context.set_context(mode=mode, device_target="Ascend")
    with no_init_parameters():
        param = Parameter(initializer("ones", shape=[2, 3]))
    param.set_dtype(ms.float16)
    param.init_data()
    assert param.dtype == ms.float16

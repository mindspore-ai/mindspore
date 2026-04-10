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
"""Test dropout operation"""

import pytest

from mindspore import jit, Tensor, mint, nn, context
from mindspore.ops import functional as F

from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import assert_executed_by_graph_mode
from tests.st.pi_jit.share.utils import pi_jit_with_config

context.set_context(mode=context.PYNATIVE_MODE)

jit_cfg = {'compile_with_try': False}


@pytest.mark.skip
@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='dryrun_only')
def test_mint_dropout_function():
    """
    Feature: mint.nn.functional.dropout()
    Description: Test mint dropout() function, it will be parsed by ast.
    Expectation: No exception, no graph-break.
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.dense = nn.Dense(in_channels=4, out_channels=4, has_bias=False)

        @pi_jit_with_config(jit_config=jit_cfg)
        def construct(self, x: Tensor, dropout_p: float, is_training: bool):
            x = self.dense(x)
            x = mint.nn.functional.dropout(x, dropout_p, is_training)  # dropout api will be parsed by ast.
            return x

    net = Net()
    a = F.randn(2, 4)
    o = net(a, 0.5, True)
    assert_executed_by_graph_mode(net.construct)

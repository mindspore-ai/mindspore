# Copyright 2026 Huawei Technologies Co., Ltd
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

"""Regression tests for dispatch modes with asynchronous PyBoost inputs."""

import numpy as np
import mindspore as ms
from mindspore import Tensor, mint
from mindspore.ops.auto_generate.gen_ops_prim import reshape_op
from tests.mark_utils import arg_mark
from tests.st.pynative.framework.test_ms_dispatch import OpTracer


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', mem_peak=2048.01)
def test_dispatch_mode_intercepts_generated_reshape_with_stub_input():
    """
    Feature: Ms dispatch
    Description: Pass a Tensor produced outside the mode to a generated PyBoost entry.
    Expectation: The mode records Reshape even when the input is represented by a StubNode.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    # Keep the producer's StubNode so the generated Converter must detect the active mode.
    input_tensor = mint.add(Tensor([1, 2, 3, 4], ms.float32), 1)
    tracer = OpTracer()

    with tracer:
        output = reshape_op(input_tensor, (2, 2))

    assert tracer.fwd_ops == ["Reshape"]
    np.testing.assert_array_equal(output.asnumpy(), [[2, 3], [4, 5]])

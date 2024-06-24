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
import mindspore as ms
from mindspore import ops, context
import numpy as np
import os
from tests.mark_utils import arg_mark


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
def test_pynative_multi_stream_vmm():
    """
    Feature: Pynative Multi-stream VMM.
    Description: Pynative Multi-stream VMM.
    Expectation: No exception.
    """
    os.environ["MS_ALLOC_CONF"] = "enable_vmm:true"
    context.set_context(mode=context.PYNATIVE_MODE)

    x = ms.Tensor(np.random.randn(256, 1024, 1024), dtype=ms.float32)
    s1 = ms.hal.Stream()
    with ms.hal.StreamCtx(s1):
        o0 = ops.broadcast_to(x, (12, 256, 1024, 1024))
        o0 += 1
        del o0
        ev = s1.record_event()

    ev.wait()
    o1 = ops.broadcast_to(x, (12, 256, 1024, 1024))
    o1 += 1
    ms.hal.synchronize()


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
def test_pynative_single_stream_vmm():
    """
    Feature: Pynative Single-stream VMM.
    Description: Pynative Single-stream VMM.
    Expectation: No exception.
    """
    os.environ["MS_ALLOC_CONF"] = "enable_vmm:true"
    context.set_context(mode=context.PYNATIVE_MODE)

    x = ms.Tensor(np.random.randn(256, 1024, 1024), dtype=ms.float32)

    o0 = ops.broadcast_to(x, (8, 256, 1024, 1024))
    o1 = o0 + 1
    o2 = o0 + 1
    del o0, o1, o2
    ms.hal.synchronize()

    o3 = ops.broadcast_to(x, (12, 256, 1024, 1024))
    o3 += 1
    ms.hal.synchronize()

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

"""Test PyNative mutli-stream"""

import os
import numpy as np
import mindspore as ms
from mindspore import context
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential',
          mem_peak=24.00)
def test_pynative_multi_stream():
    """
    Feature: PyNative multi-stream
    Description: Test PyNative multi-stream with memory reuse.
    Expectation: run success
    """

    x_np = np.ones((1024, 1024)).astype(np.float32)
    x = ms.Tensor.from_numpy(x_np)
    s1 = ms.runtime.Stream()
    for _ in range(100):
        with ms.runtime.StreamCtx(s1):
            y = x + 1
            z = ms.mint.matmul(y, y)
            event = s1.record_event()

            # Free tensor memory
            del y

        # Execute on default stream.
        # Memory reuse is prevented as different streams utilize separate memory pools.
        empty = ms.mint.empty_like(x)
        zeros = ms.mint.zeros_like(x)
        empty.copy_(zeros)

        cur_stream = ms.runtime.current_stream()
        cur_stream.wait_event(event)

        np.allclose(z.asnumpy(), np.matmul(x_np + 1, x_np + 1))


def test_pynative_aclop_multi_stream():
    """
    Feature: PyNative multi-stream
    Description: Test PyNative multi-stream with aclop cache hit.
    Expectation: run success
    """
    context.set_context(mode=context.GRAPH_MODE)

    x = np.ones((192,), dtype=np.float32)
    # data on Device with stream 0
    a = ms.from_numpy(x).sin()

    # data on Device with stream 3
    s1 = ms.runtime.Stream()
    with ms.runtime.StreamCtx(s1):
        b = ms.from_numpy(x).sin()
    ms.runtime.synchronize()

    # data on Device with stream 0
    c = ms.from_numpy(x)

    ms.ops.Identity()(a)
    ms.ops.Identity()(b)
    ms.ops.Identity()(c)

    ms.runtime.synchronize()


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level0',
          card_mark='allcards',
          essential_mark='essential')
def test_pynative_communication_multi_stream_memory():
    '''
    Feature: Memory with multi-stream.
    Description: Test PyNative communication memory with multi-stream.
    Expectation: Run success.
    '''
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 --master_port=10969 --join=True " \
        "pytest -s test_comm_op_memory.py::test_communication_op_with_stream"
    )
    assert return_code == 0

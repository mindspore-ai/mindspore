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
"""Test mindspore.hal"""

import numpy as np
import mindspore as ms
from mindspore import Tensor, jit, ops
from mindspore.nn import Cell

from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import match_array
from tests.st.pi_jit.one_stage.test_utils import save_graph_ir, check_ir_num


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_hal_stream_ctx_matmul():
    """
    Feature: mindspore.hal stream context under PIJit.
    Description: Execute matmul inside StreamCtx, synchronize explicitly and compare PIJit with pynative mode.
    Expectation: JIT result matches pynative result and generates two graphs.
    Migrated from: test_pijit_cfunc_buildin.py::test_pijit_func_split_only_steam
    """

    class Net(Cell):
        def construct(self, a, b):
            stream = ms.hal.Stream()
            with ms.hal.StreamCtx(stream):
                c = ops.matmul(a, b)
            out = a + b
            stream.synchronize()
            return out * c

    np_a = np.ones((20, 20), np.float32)
    np_b = np.ones((20, 20), np.float32)
    tensor_a = Tensor(np_a, ms.float32)
    tensor_b = Tensor(np_b, ms.float32)

    pynative_net = Net()
    pynative_out = pynative_net(tensor_a, tensor_b)

    jit_net = Net()
    jit_net.construct = jit(jit_net.construct, capture_mode='bytecode')
    jit_out = jit_net(tensor_a, tensor_b)

    match_array(pynative_out, jit_out)
    check_ir_num('graph_before_compile', 2)

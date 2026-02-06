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
"""Test mindspore.dataset API"""

import numpy as np
import mindspore as ms
from mindspore import Tensor, jit
from mindspore.nn import Cell
from mindspore.dataset.core.datatypes import mstype_to_detype

from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dataset_dtype_to_nptype_constant():
    """
    Feature: Dataset dtype conversion utility in PIJit.
    Description: Convert Tensor dtype to dataset DataType within jit-compiled Cell and compare with pynative result.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_cfunc_buildin.py::test_pijit_compile_constant_dtype_to_nptype
    """

    class Net(Cell):
        @jit(capture_mode='bytecode', fullgraph=True)
        def construct(self, x):
            return mstype_to_detype(x.dtype)

    input_np = np.random.rand(2, 2, 3).astype(np.float32)
    input_x = Tensor(input_np)

    net = Net()
    out = net(input_x)
    assert out == ms._c_dataengine.DataType("float32")

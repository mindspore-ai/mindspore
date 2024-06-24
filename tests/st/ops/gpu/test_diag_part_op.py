# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark

import numpy as np
import pytest
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops import operations as P


def diag_part(nptype):
    x_np = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(nptype))
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    output_ms = P.DiagPart()(Tensor(x_np))
    output_expect = np.array([1, 5, 9])
    assert np.allclose(output_ms.asnumpy(), output_expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_diag_part_float16():
    """
    Feature: DiagPart op.
    Description: Test diag_part op with 3d and float16.
    Expectation: The value and shape of output are the expected values.
    """
    diag_part(np.float16)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_diag_part_float32():
    """
    Feature: DiagPart op.
    Description: Test diag_part op with 3d and float32.
    Expectation: The value and shape of output are the expected values.
    """
    diag_part(np.float32)

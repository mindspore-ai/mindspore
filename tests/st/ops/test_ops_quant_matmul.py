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

import numpy as np
import pytest
import mindspore as ms
from mindspore import ops, Tensor
from mindspore.graph.api import _pynative_executor


# ascend backend can't support fp8 dtypes on 910b, fp8 dtypes need A5 now.
# @arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize('dtype', ["fp8_e4m3fn", "fp8_e5m2", "hifp8"])
def test_ops_quant_matmul_fp8(dtype):
    """
    Feature: ops.quant_matmul.
    Description: test function with fp8 dtypes
    Expectation: expect correct errors.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)

    try:
        from mindspore import float8_e4m3fn, float8_e5m2, hifloat8
        dtypes = {
            "fp8_e4m3fn": float8_e4m3fn,
            "fp8_e5m2": float8_e5m2,
            "hifp8": hifloat8,
        }
        dtype = dtypes[dtype]
    except ImportError:
        dtype = ms.float16

    x1_np = np.random.randn(2, 3, 4).astype(np.float32)
    x2_np = np.random.randn(2, 4, 5).astype(np.float32)
    scale_np = np.random.randn(1,).astype(np.float32)
    pertoken_scale_np = np.random.randn(3,).astype(np.float32)

    x1 = Tensor(x1_np, dtype)
    x2 = Tensor(x2_np, dtype)
    scale = Tensor(scale_np)
    pertoken_scale = Tensor(pertoken_scale_np)

    try:
        out = ops.auto_generate.quant_matmul(x1, x2, scale, pertoken_scale=pertoken_scale, output_dtype=ms.bfloat16)
        _pynative_executor.sync()
        assert out.shape == (2, 3, 5)
        assert out.dtype == ms.bfloat16
    # pylint: disable=broad-except
    except Exception as error:
        if "aclnnQuantMatmulV5GetWorkspaceSize not in" not in str(error):
            print(error)
            assert False

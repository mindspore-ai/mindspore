# Copyright 2021 Huawei Technologies Co., Ltd
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
"""st for scipy.linalg."""

import pytest
import numpy as onp
import scipy as osp

from mindspore import Tensor
import mindspore.scipy as msp
from .utils import match_array, create_full_rank_matrix

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('args', [(), (1,), (7, -1), (3, 4, 5),
                                  (onp.ones((3, 4), dtype=onp.float32), 5, onp.random.randn(5, 2).astype(onp.float32))])
def test_block_diag(args):
    """
    Feature: ALL TO ALL
    Description: test cases for block_diag
    Expectation: the result match scipy
    """
    tensor_args = tuple([Tensor(arg) for arg in args])
    ms_res = msp.linalg.block_diag(*tensor_args)

    scipy_res = osp.linalg.block_diag(*args)
    match_array(ms_res.asnumpy(), scipy_res)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64])
@pytest.mark.parametrize('shape', [(4, 4), (50, 50), (2, 5, 5)])
def test_inv(dtype, shape):
    """
    Feature: ALL TO ALL
    Description: test cases for inv
    Expectation: the result match numpy
    """
    onp.random.seed(0)
    x = create_full_rank_matrix(shape, dtype)

    ms_res = msp.linalg.inv(Tensor(x))
    scipy_res = onp.linalg.inv(x)
    match_array(ms_res.asnumpy(), scipy_res, error=3)

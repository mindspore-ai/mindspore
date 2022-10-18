# Copyright 2022 Huawei Technologies Co., Ltd
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

from mindspore.common.api import jit
from mindspore.common.api import _pynative_executor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.functional import vmap
from mindspore import Tensor
from mindspore import context


def np_all_close_with_loss(out, expect):
    """np_all_close_with_loss"""
    return np.allclose(out, expect, 0.0005, 0.0005, equal_nan=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("data_type", [np.int32, np.int64])
def test_data_formata_dim_map_gpu(data_type):
    """
    Feature: DataFormatDimMapNet gpu kernel.
    Description: test the rightness of DataFormatDimMapNet gpu kernel.
    Expectation: Success.
    """
    x_np_1_gpu = np.array([-4, -3, -2, -1, 0, 1, 2, 3]).astype(data_type)
    output_1_gpu = P.DataFormatDimMap()(Tensor(x_np_1_gpu))
    output_1_expect_gpu = np.array([0, 3, 1, 2, 0, 3, 1, 2]).astype(data_type)
    assert np.allclose(output_1_gpu.asnumpy(), output_1_expect_gpu)

    output_2_gpu = P.DataFormatDimMap(src_format="NHWC", dst_format="NHWC")(Tensor(x_np_1_gpu))
    output_2_expect_gpu = np.array([0, 1, 2, 3, 0, 1, 2, 3]).astype(data_type)
    assert np.allclose(output_2_gpu.asnumpy(), output_2_expect_gpu)

    output_3_gpu = P.DataFormatDimMap(src_format="NCHW", dst_format="NHWC")(Tensor(x_np_1_gpu))
    output_3_expect_gpu = np.array([0, 2, 3, 1, 0, 2, 3, 1]).astype(data_type)
    assert np.allclose(output_3_gpu.asnumpy(), output_3_expect_gpu)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("data_type", [np.int32, np.int64])
def test_data_formata_dim_map_vmap_gpu(data_type):
    """
    Feature: DataFormatDimMapNet gpu kernel
    Description: test the rightness of DataFormatDimMapNet gpu kernel vmap feature.
    Expectation: Success.
    """

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    def data_formata_dim_map_fun_gpu(x):
        """data_formata_dim_map_fun_gpu"""
        return P.DataFormatDimMap()(x)

    x_np_gpu = np.random.randint(low=-4, high=4, size=(100, 100)).astype(data_type)
    x_gpu = Tensor(x_np_gpu)
    x_gpu = F.sub(x_gpu, 0)

    output_vmap_gpu = vmap(data_formata_dim_map_fun_gpu, in_axes=(0,))(x_gpu)
    _pynative_executor.sync()

    @jit
    def manually_batched_gpu(xs):
        """manually_batched_gpu"""
        output_gpu = []
        for i in range(xs.shape[0]):
            output_gpu.append(data_formata_dim_map_fun_gpu(xs[i]))
        return F.stack(output_gpu)

    output_manually_gpu = manually_batched_gpu(x_gpu)
    _pynative_executor.sync()

    assert np_all_close_with_loss(output_vmap_gpu.asnumpy(), output_manually_gpu.asnumpy())

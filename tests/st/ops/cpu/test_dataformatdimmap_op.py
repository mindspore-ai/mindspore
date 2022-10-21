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
from mindspore import Tensor
from mindspore import context
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.functional import vmap
from mindspore.common.api import jit
from mindspore.common.api import _pynative_executor


def np_all_close_with_loss(out, expect):
    """np_all_close_with_loss"""
    return np.allclose(out, expect, 0.0005, 0.0005, equal_nan=True)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("data_type", [np.int32, np.int64])
def test_data_formata_dim_map(data_type):
    """
    Feature: DataFormatDimMapNet cpu kernel.
    Description: test the rightness of DataFormatDimMapNet cpu kernel.
    Expectation: Success.
    """
    x_np_1 = np.array([-4, -3, -2, -1, 0, 1, 2, 3]).astype(data_type)
    output_1 = P.DataFormatDimMap()(Tensor(x_np_1))
    output_1_expect = np.array([0, 3, 1, 2, 0, 3, 1, 2]).astype(data_type)
    assert np.allclose(output_1.asnumpy(), output_1_expect)

    output_2 = P.DataFormatDimMap(src_format="NHWC", dst_format="NHWC")(Tensor(x_np_1))
    output_2_expect = np.array([0, 1, 2, 3, 0, 1, 2, 3]).astype(data_type)
    assert np.allclose(output_2.asnumpy(), output_2_expect)

    output_3 = P.DataFormatDimMap(src_format="NCHW", dst_format="NHWC")(Tensor(x_np_1))
    output_3_expect = np.array([0, 2, 3, 1, 0, 2, 3, 1]).astype(data_type)
    assert np.allclose(output_3.asnumpy(), output_3_expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("data_type", [np.int32, np.int64])
def test_data_formata_dim_map_vmap(data_type):
    """
    Feature: DataFormatDimMapNet cpu kernel
    Description: test the rightness of DataFormatDimMapNet cpu kernel vmap feature.
    Expectation: Success.
    """

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    def data_formata_dim_map_fun(x):
        """data_formata_dim_map_fun"""
        return P.DataFormatDimMap()(x)

    x_np = np.random.randint(low=-4, high=4, size=(100, 100)).astype(data_type)
    x = Tensor(x_np)
    x = F.sub(x, 0)

    output_vmap = vmap(data_formata_dim_map_fun, in_axes=(0,))(x)
    _pynative_executor.sync()

    @jit
    def manually_batched(xs):
        """manually_batched"""
        output = []
        for i in range(xs.shape[0]):
            output.append(data_formata_dim_map_fun(xs[i]))
        return F.stack(output)

    output_manually = manually_batched(x)
    _pynative_executor.sync()

    assert np_all_close_with_loss(output_vmap.asnumpy(), output_manually.asnumpy())

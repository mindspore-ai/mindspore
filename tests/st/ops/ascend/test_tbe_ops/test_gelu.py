# Copyright 2020 Huawei Technologies Co., Ltd
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
import math
import numpy as np
import pytest

from mindspore import context
from mindspore import log as logger
from mindspore.common.tensor import Tensor
from mindspore.nn import GELU, Cell
from mindspore.ops import operations as P
from mindspore.train.model import Model

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


def gelu_forward_me_impl(input):
    n = GELU()
    n.set_train()
    m = Model(n)
    out = m.predict(input)
    return out.asnumpy()


def gelu_forward_cmp(input_shape, data_type=np.float32):
    input_np = np.random.randn(*input_shape).astype(data_type)
    input_me = Tensor(input_np)
    out_me = gelu_forward_me_impl(input_me)


@pytest.mark.skip(reason="scalar")
def test_gelu_input_scalar():
    input_np = np.random.randn()
    input_me = Tensor(input_np)
    out_me = gelu_forward_me_impl(input_me)
    logger.info("---------me--------")
    logger.info(out_me)


def test_gelu_input_dim_0():
    input_shape = [0]
    with pytest.raises(ValueError):
        gelu_forward_cmp(input_shape)


def test_gelu_input_dim_10240_1024():
    input_shape = [10240, 1024]
    gelu_forward_cmp(input_shape)


def test_gelu_input_dim_10240_768():
    input_shape = [10240, 768]
    gelu_forward_cmp(input_shape)


@pytest.mark.lower_bs
def test_gelu_input_dim_1024_3072():
    input_shape = [1024, 3072]
    gelu_forward_cmp(input_shape)


@pytest.mark.lower_bs
def test_gelu_input_dim_1024_4096():
    input_shape = [1024, 4096]
    gelu_forward_cmp(input_shape)


def test_gelu_input_dim_1280_1024():
    input_shape = [1280, 1024]
    gelu_forward_cmp(input_shape)


@pytest.mark.lower_bs
def test_gelu_input_dim_1280_768():
    input_shape = [1280, 768]
    gelu_forward_cmp(input_shape)


@pytest.mark.lower_bs
def test_gelu_input_dim_128_3072():
    input_shape = [128, 3072]
    gelu_forward_cmp(input_shape)


@pytest.mark.lower_bs
def test_gelu_input_dim_128_4096():
    input_shape = [128, 4096]
    gelu_forward_cmp(input_shape)


@pytest.mark.lower_bs
def test_gelu_input_dim_160_1024():
    input_shape = [160, 1024]
    gelu_forward_cmp(input_shape)


@pytest.mark.lower_bs
def test_gelu_input_dim_160_768():
    input_shape = [160, 768]
    gelu_forward_cmp(input_shape)


@pytest.mark.lower_bs
def test_gelu_input_dim_16384_3072():
    input_shape = [16384, 3072]
    gelu_forward_cmp(input_shape)


def test_gelu_input_dim_16384_4096():
    input_shape = [16384, 4096]
    gelu_forward_cmp(input_shape)


@pytest.mark.lower_bs
def test_gelu_input_dim_20_1024():
    input_shape = [20, 1024]
    gelu_forward_cmp(input_shape)


def test_gelu_input_dim_20480_1024():
    input_shape = [20480, 1024]
    gelu_forward_cmp(input_shape)

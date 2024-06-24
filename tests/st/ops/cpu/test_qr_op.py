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
from tests.mark_utils import arg_mark
import pytest
import numpy as onp
import scipy as osp
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import _checkparam as validator
from mindspore.common import dtype as mstype
from mindspore.ops import PrimitiveWithInfer
from mindspore.ops import prim_attr_register

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class QR(PrimitiveWithInfer):
    """
    QR decomposition
    A = Q.R
    """

    @prim_attr_register
    def __init__(self, mode: str = "full"):
        super().__init__(name="QR")
        self.mode = validator.check_value_type("mode", mode, [str], self.name)

        self.init_prim_io_names(inputs=['x'], outputs=['q', 'r'])

    def __infer__(self, x):
        x_shape = x['shape']
        x_dtype = x['dtype']
        m, n = x_shape
        if self.mode == "economic":
            q_shape = (m, min(m, n))
            r_shape = (min(m, n), n)
        else:
            q_shape = (m, m)
            r_shape = (m, n)

        output = {
            'shape': (q_shape, r_shape),
            'dtype': (x_dtype, x_dtype),
            'value': None
        }
        return output

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid(x_dtype, [mstype.float32, mstype.float64], self.name, True)
        return x_dtype


def _match_array(actual, expected, error=0):
    if isinstance(actual, int):
        actual = onp.asarray(actual)
    if isinstance(actual, tuple):
        actual = onp.asarray(actual)

    if error > 0:
        onp.testing.assert_almost_equal(actual, expected, decimal=error)
    else:
        onp.testing.assert_equal(actual, expected)


class QRNet(nn.Cell):
    def __init__(self, mode: str = "full"):
        super(QRNet, self).__init__()
        self.mode = mode
        self.qr = QR(mode=self.mode)

    def construct(self, a):
        q, r = self.qr(a)
        if self.mode == 'r':
            return (r,)
        return q, r


@pytest.mark.parametrize('a_shape', [(9, 6), (6, 9)])
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64])
@pytest.mark.parametrize('mode', ['full', 'r', 'economic'])
def test_lu_net(a_shape, dtype, mode):
    """
    Feature: ALL To ALL
    Description: test cases for qr decomposition test cases for A = Q.R
    Expectation: the result match to scipy
    """
    onp.random.seed(0)

    if mode == 'r':
        m, n = a_shape
        a = onp.random.random((m, n)).astype(dtype)
        osp_r = osp.linalg.qr(a, mode=mode)

        msp_qr = QRNet(mode=mode)
        tensor_a = Tensor(a)
        msp_r = msp_qr(tensor_a)

        _match_array(msp_r[0].asnumpy(), osp_r[0], error=5)
    else:
        m, n = a_shape
        a = onp.random.random((m, n)).astype(dtype)
        osp_q, osp_r = osp.linalg.qr(a, mode=mode)

        msp_qr = QRNet(mode=mode)
        tensor_a = Tensor(a)
        msp_q, msp_r = msp_qr(tensor_a)

        _match_array(msp_q.asnumpy(), osp_q, error=5)
        _match_array(msp_r.asnumpy(), osp_r, error=5)

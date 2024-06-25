# Copyright 2023 Huawei Technologies Co., Ltd
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
import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor, context
from mindspore import ops
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def qr_forward_func(x, full_matrices):
    return ops.Qr(full_matrices)(x)


@test_utils.run_with_cell
def qr_backward_func(x, full_matrices):
    return ops.grad(qr_forward_func, (0, 1))(x, full_matrices)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_qr_forward(mode):
    """
    Feature: qr ops.
    Description: test ops qr.
    Expectation: output right results.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[20., -31, 7],
                         [4, 270, -90],
                         [-8, 17, -32]]).astype(np.float32))
    output_q, output_r = qr_forward_func(x, False)
    print("output_r:\n", output_r)
    expect_output_q = np.asarray([[-0.912871, 0.16366126, 0.37400758],
                                  [-0.18257418, -0.9830709, -0.01544376],
                                  [0.36514837, -0.08238228, 0.92729706]]).astype(np.float32)
    expect_output_r = np.asarray([[-21.908903, -14.788506, -1.6431675],
                                  [0., -271.9031, 92.25824],
                                  [0., 0., -25.665514]]).astype(np.float32)
    assert np.allclose(output_q.asnumpy(), expect_output_q)
    assert np.allclose(output_r.asnumpy(), expect_output_r)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_qr_dynamic(mode):
    """
    Feature: qr ops.
    Description: test ops qr dynamic tensor input.
    Expectation: output right results.
    """
    def qr_(x, full_matrices):
        return ops.Qr(full_matrices)(x)

    context.set_context(mode=mode)
    x_dyn = Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(qr_)
    test_cell.set_inputs(x_dyn, False)
    x1 = Tensor(np.array([[1., 2., 3.],
                          [4., 5., 6.]]).astype(np.float32))
    output1_q, output1_r = test_cell(x1, False)
    print("output1_q:\n", output1_q)
    print("output1_r:\n", output1_r)
    expect_output1_q = np.asarray([[-0.24253559, -0.97014254],
                                   [-0.97014254, 0.24253553]]).astype(np.float32)
    expect_output1_r = np.asarray([[-4.1231055, -5.3357835, -6.548462],
                                   [0., -0.72760725, -1.455214]]).astype(np.float32)
    assert np.allclose(output1_q.asnumpy(), expect_output1_q)
    assert np.allclose(output1_r.asnumpy(), expect_output1_r)
    x2 = Tensor(np.array([[1., 2.],
                          [3., 4.]]).astype(np.float32))
    output2_q, output2_r = test_cell(x2, False)
    print("output2_q:\n", output2_q)
    print("output2_r:\n", output2_r)
    expect_output2_q = np.asarray([[-0.3162278, -0.9486833],
                                   [-0.9486833, 0.31622773]]).astype(np.float32)
    expect_output2_r = np.asarray([[-3.1622777, -4.4271884],
                                   [0., -0.63245535]]).astype(np.float32)
    assert np.allclose(output2_q.asnumpy(), expect_output2_q)
    assert np.allclose(output2_r.asnumpy(), expect_output2_r)

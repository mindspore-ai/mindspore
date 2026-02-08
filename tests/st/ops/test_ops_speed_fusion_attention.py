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
"""Speed fusion attention test cases."""
import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore.ops import speed_fusion_attention
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.common.random_generator import generate_numpy_ndarray_by_randn


@test_utils.run_with_cell
def speed_fusion_attention_forward_func(query, key, value, head_num, input_layout, pse=None, scale=1.0, keep_prob=1.0,
                                        atten_mask=None, pre_tokens=2147483647, next_tokens=2147483647,
                                        actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0, pse_type=1):
    return speed_fusion_attention(query, key, value, head_num, input_layout, pse=pse, scale=scale, keep_prob=keep_prob,
                                  atten_mask=atten_mask, pre_tokens=pre_tokens, next_tokens=next_tokens,
                                  actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen,
                                  sparse_mode=sparse_mode, pse_type=pse_type)


@test_utils.run_with_cell
def speed_fusion_attention_backward_func(query, key, value, head_num, input_layout, pse=None, scale=1.0, keep_prob=1.0,
                                         atten_mask=None, pre_tokens=2147483647, next_tokens=2147483647,
                                         actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0, pse_type=1):
    return ms.grad(speed_fusion_attention_forward_func, (0, 1, 2, 5))(query, key, value, head_num, input_layout, pse,
                                                                      scale, keep_prob, atten_mask, pre_tokens,
                                                                      next_tokens, actual_seq_qlen, actual_seq_kvlen,
                                                                      sparse_mode, pse_type)


def generate_random_input(shape, pse_type):
    query = generate_numpy_ndarray_by_randn(shape, np.float16, 'query')
    key = generate_numpy_ndarray_by_randn(shape, np.float16, 'key')
    value = generate_numpy_ndarray_by_randn(shape, np.float16, 'value')
    pse = generate_numpy_ndarray_by_randn(shape, np.float16, 'pse')
    if pse_type > 1:
        pse = np.array([0.0625, 0.0039]).astype(np.float32)
    return query, key, value, pse


def get_atten_mask(shape, pre_tokens, next_tokens, sparse_mode):
    atten_mask = None
    if sparse_mode == 0:
        atten_mask_u = np.triu(np.ones(shape), k=pre_tokens + 1)
        atten_mask_l = np.tril(np.ones(shape), k=-next_tokens - 1)
        atten_mask = atten_mask_u + atten_mask_l
        atten_mask = atten_mask.astype(np.bool_)
    elif sparse_mode == 1:
        atten_mask = np.zeros(shape).astype(np.bool_)
    elif sparse_mode == 2 or sparse_mode == 3 or sparse_mode == 4:
        atten_mask = np.triu(np.ones((2048, 2048)), k=1)
        atten_mask = atten_mask.astype(np.bool_)
    return atten_mask


def speed_fusion_attention_case0():
    """
    Feature: Test the precision for TND with pse_type is 0 and sparse_mode is 0.
    Description: Test function speed_fusion_attention forward and backward.
    Expectation: The result is correct.
    """
    shape = (2, 4, 4)
    head_num = 4
    input_layout = "TND"
    sparse_mode = 0
    pre_tokens = 2
    next_tokens = 2
    scale = 0.1
    keep_prob = 0.3
    pse_type = 0
    actual_seq_qlen = (2, 2, 2)
    actual_seq_kvlen = (2, 2, 2)

    query_np, key_np, value_np, _ = generate_random_input(shape, pse_type)

    ms.manual_seed(10)
    grads = speed_fusion_attention_backward_func(Tensor(query_np), Tensor(key_np), Tensor(value_np), head_num,
                                                 input_layout, pse=None, scale=scale, keep_prob=keep_prob,
                                                 atten_mask=None, pre_tokens=pre_tokens, next_tokens=next_tokens,
                                                 actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen,
                                                 sparse_mode=sparse_mode, pse_type=pse_type)
    expect_grad = np.zeros((2, 4, 4)).astype(np.float16)
    np.testing.assert_allclose(grads[0].asnumpy(), expect_grad, rtol=1e-3)
    np.testing.assert_allclose(grads[1].asnumpy(), expect_grad, rtol=1e-3)
    np.testing.assert_allclose(grads[2].asnumpy(), expect_grad, rtol=1e-3)

    atten_out = np.array([[[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]],
                          [[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]]]).astype(np.float16)
    softmax_max = np.array(
        [[[0.48339483, 0.48339483, 0.48339483, 0.48339483, 0.48339483, 0.48339483, 0.48339483, 0.48339483],
          [0.10324126, 0.10324126, 0.10324126, 0.10324126, 0.10324126, 0.10324126, 0.10324126, 0.10324126],
          [0.03119416, 0.03119416, 0.03119416, 0.03119416, 0.03119416, 0.03119416, 0.03119416, 0.03119416],
          [0.27002069, 0.27002069, 0.27002069, 0.27002069, 0.27002069, 0.27002069, 0.27002069, 0.27002069]],
         [[0.49907133, 0.49907133, 0.49907133, 0.49907133, 0.49907133, 0.49907133, 0.49907133, 0.49907133],
          [0.07635707, 0.07635707, 0.07635707, 0.07635707, 0.07635707, 0.07635707, 0.07635707, 0.07635707],
          [0.26141486, 0.26141486, 0.26141486, 0.26141486, 0.26141486, 0.26141486, 0.26141486, 0.26141486],
          [0.48785314, 0.48785314, 0.48785314, 0.48785314, 0.48785314, 0.48785314, 0.48785314, 0.48785314]]]
        ).astype(np.float32)
    softmax_sum = np.array(
        [[[1.61259103, 1.61259103, 1.61259103, 1.61259103, 1.61259103, 1.61259103, 1.61259103, 1.61259103],
          [1.89592004, 1.89592004, 1.89592004, 1.89592004, 1.89592004, 1.89592004, 1.89592004, 1.89592004],
          [1.90331483, 1.90331483, 1.90331483, 1.90331483, 1.90331483, 1.90331483, 1.90331483, 1.90331483],
          [1.71140695, 1.71140695, 1.71140695, 1.71140695, 1.71140695, 1.71140695, 1.71140695, 1.71140695]],
         [[1.58519340, 1.58519340, 1.58519340, 1.58519340, 1.58519340, 1.58519340, 1.58519340, 1.58519340],
          [1.89306259, 1.89306259, 1.89306259, 1.89306259, 1.89306259, 1.89306259, 1.89306259, 1.89306259],
          [1.94897914, 1.94897914, 1.94897914, 1.94897914, 1.94897914, 1.94897914, 1.94897914, 1.94897914],
          [1.75668609, 1.75668609, 1.75668609, 1.75668609, 1.75668609, 1.75668609, 1.75668609, 1.75668609]]]
        ).astype(np.float32)

    ms.manual_seed(10)
    out = speed_fusion_attention_forward_func(Tensor(query_np), Tensor(key_np), Tensor(value_np), head_num,
                                              input_layout, pse=None, scale=scale, keep_prob=keep_prob, atten_mask=None,
                                              pre_tokens=pre_tokens, next_tokens=next_tokens,
                                              actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen,
                                              sparse_mode=sparse_mode, pse_type=pse_type)
    np.testing.assert_allclose(out[0].asnumpy(), atten_out, rtol=1e-3)
    np.testing.assert_allclose(out[1].asnumpy(), softmax_max, rtol=1e-4)
    np.testing.assert_allclose(out[2].asnumpy(), softmax_sum, rtol=1e-4)
    assert out[3].shape == (0,)
    assert out[4] == Tensor(10, dtype=ms.int64)
    assert out[5] == Tensor(0, dtype=ms.int64)
    assert out[6] == Tensor(16, dtype=ms.int64)

    assert out[0].dtype == ms.float16
    assert out[1].dtype == ms.float32
    assert out[2].dtype == ms.float32
    assert out[3].dtype == ms.float16
    assert out[4].dtype == ms.int64
    assert out[5].dtype == ms.int64
    assert out[6].dtype == ms.int64


def speed_fusion_attention_case1():
    """
    Feature: Test the precision for BNSD with pse_type is 1 and sparse_mode is 1.
    Description: Test function speed_fusion_attention forward and backward.
    Expectation: The result is correct.
    """
    shape = (1, 2, 4, 4)
    head_num = 2
    input_layout = "BNSD"
    sparse_mode = 1
    pre_tokens = 8
    next_tokens = 8
    scale = 0.2
    keep_prob = 0.5
    pse_type = 1

    query_np, key_np, value_np, pse_np = generate_random_input(shape, pse_type)
    atten_mask_np = get_atten_mask(shape, pre_tokens, next_tokens, sparse_mode)

    ms.manual_seed(10)
    grads = speed_fusion_attention_backward_func(Tensor(query_np), Tensor(key_np), Tensor(value_np), head_num,
                                                 input_layout, pse=Tensor(pse_np), scale=scale, keep_prob=keep_prob,
                                                 atten_mask=Tensor(atten_mask_np), pre_tokens=pre_tokens,
                                                 next_tokens=next_tokens, sparse_mode=sparse_mode, pse_type=pse_type)

    dq = np.array([[[[6.72851562e-01, 4.03564453e-01, 2.89062500e-01, -3.28979492e-02],
                     [1.06054688e+00, 5.52246094e-01, 6.31835938e-01, 3.06640625e-01],
                     [-3.63159180e-02, 5.87768555e-02, -9.70458984e-02, -5.59997559e-02],
                     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]],
                    [[1.78710938e-01, 3.12255859e-01, -9.37500000e-02, 5.52673340e-02],
                     [-7.73315430e-02, -1.64428711e-01, 2.01416016e-01, -4.19311523e-02],
                     [-2.83660889e-02, -8.71276855e-03, 9.54627991e-04, -2.66265869e-02],
                     [1.11206055e-01, 5.91125488e-02, 1.30981445e-01, 6.41479492e-02]]]]).astype(np.float16)
    dk = np.array([[[[0.39599609, 0.10839844, 0.13928223, -0.16833496],
                     [-0.26098633, -0.10711670, -0.14782715, 0.01760864],
                     [-0.05023193, -0.02578735, 0.05444336, 0.03341675],
                     [-0.08477783, 0.02449036, -0.04583740, 0.11718750]],
                    [[-0.05215454, 0.01041412, -0.02619934, -0.07940674],
                     [0.02288818, -0.15478516, 0.02853394, 0.12939453],
                     [-0.06146240, 0.07659912, -0.28271484, -0.07482910],
                     [0.09075928, 0.06781006, 0.28002930, 0.02490234]]]]).astype(np.float16)
    dv = np.array([[[[1.63476562, 1.63476562, 1.63476562, 1.63476562],
                     [0.00000000, 0.00000000, 0.00000000, 0.00000000],
                     [0.58935547, 0.58935547, 0.58935547, 0.58935547],
                     [0.46826172, 0.46826172, 0.46826172, 0.46826172]],
                    [[0.73535156, 0.73535156, 0.73535156, 0.73535156],
                     [0.82519531, 0.82519531, 0.82519531, 0.82519531],
                     [0.95800781, 0.95800781, 0.95800781, 0.95800781],
                     [1.21289062, 1.21289062, 1.21289062, 1.21289062]]]]).astype(np.float16)
    np.testing.assert_allclose(grads[0].asnumpy(), dq, rtol=1e-3)
    np.testing.assert_allclose(grads[1].asnumpy(), dk, rtol=1e-3)
    np.testing.assert_allclose(grads[2].asnumpy(), dv, rtol=1e-3)

    atten_out = np.array([[[[2.30859375, 0.76806641, 1.14648438, -0.56835938],
                            [-0.10369873, -0.44897461, -0.13220215, -0.78466797],
                            [-0.15234375, -0.67236328, 0.31250000, -0.02047729],
                            [0.00000000, 0.00000000, 0.00000000, 0.00000000]],
                           [[0.53320312, 0.64843750, 0.11901855, 0.30737305],
                            [0.22900391, -0.20507812, 0.87695312, 0.35717773],
                            [-0.06768799, -0.02088928, 0.51123047, -0.07952881],
                            [0.50439453, -0.07922363, 1.66210938, 0.44653320]]]]).astype(np.float16)
    softmax_max = np.array(
        [[[[1.33612561, 1.33612561, 1.33612561, 1.33612561, 1.33612561, 1.33612561, 1.33612561, 1.33612561],
           [0.15099691, 0.15099691, 0.15099691, 0.15099691, 0.15099691, 0.15099691, 0.15099691, 0.15099691],
           [0.83408016, 0.83408016, 0.83408016, 0.83408016, 0.83408016, 0.83408016, 0.83408016, 0.83408016],
           [0.51408333, 0.51408333, 0.51408333, 0.51408333, 0.51408333, 0.51408333, 0.51408333, 0.51408333]],
          [[0.29119930, 0.29119930, 0.29119930, 0.29119930, 0.29119930, 0.29119930, 0.29119930, 0.29119930],
           [0.77636951, 0.77636951, 0.77636951, 0.77636951, 0.77636951, 0.77636951, 0.77636951, 0.77636951],
           [0.33583462, 0.33583462, 0.33583462, 0.33583462, 0.33583462, 0.33583462, 0.33583462, 0.33583462],
           [1.00331867, 1.00331867, 1.00331867, 1.00331867, 1.00331867, 1.00331867, 1.00331867, 1.00331867]]]]
        ).astype(np.float32)
    softmax_sum = np.array(
        [[[[1.59921753, 1.59921753, 1.59921753, 1.59921753, 1.59921753, 1.59921753, 1.59921753, 1.59921753],
           [3.39330482, 3.39330482, 3.39330482, 3.39330482, 3.39330482, 3.39330482, 3.39330482, 3.39330482],
           [2.08144379, 2.08144379, 2.08144379, 2.08144379, 2.08144379, 2.08144379, 2.08144379, 2.08144379],
           [2.87661767, 2.87661767, 2.87661767, 2.87661767, 2.87661767, 2.87661767, 2.87661767, 2.87661767]],
          [[3.14441872, 3.14441872, 3.14441872, 3.14441872, 3.14441872, 3.14441872, 3.14441872, 3.14441872],
           [2.64547515, 2.64547515, 2.64547515, 2.64547515, 2.64547515, 2.64547515, 2.64547515, 2.64547515],
           [3.30701256, 3.30701256, 3.30701256, 3.30701256, 3.30701256, 3.30701256, 3.30701256, 3.30701256],
           [2.45223570, 2.45223570, 2.45223570, 2.45223570, 2.45223570, 2.45223570, 2.45223570, 2.45223570]]]]
        ).astype(np.float32)

    ms.manual_seed(10)
    out = speed_fusion_attention_forward_func(Tensor(query_np), Tensor(key_np), Tensor(value_np), head_num,
                                              input_layout, pse=Tensor(pse_np), scale=scale, keep_prob=keep_prob,
                                              atten_mask=Tensor(atten_mask_np), pre_tokens=pre_tokens,
                                              next_tokens=next_tokens, sparse_mode=sparse_mode, pse_type=pse_type)
    np.testing.assert_allclose(out[0].asnumpy(), atten_out, rtol=1e-3)
    np.testing.assert_allclose(out[1].asnumpy(), softmax_max, rtol=1e-4)
    np.testing.assert_allclose(out[2].asnumpy(), softmax_sum, rtol=1e-4)
    assert out[3].shape == (0,)
    assert out[4] == Tensor(10, dtype=ms.int64)
    assert out[5] == Tensor(0, dtype=ms.int64)
    assert out[6] == Tensor(32, dtype=ms.int64)


def speed_fusion_attention_case2():
    """
    Feature: Test the precision for BNSD with pse_type is 2 and sparse_mode is 2.
    Description: Test function speed_fusion_attention forward and backward.
    Expectation: The result is correct.
    """
    shape = (1, 2, 4, 4)
    head_num = 2
    input_layout = "BNSD"
    sparse_mode = 2
    pre_tokens = 8
    next_tokens = 0
    scale = 0.3
    keep_prob = 0.7
    pse_type = 2

    query_np, key_np, value_np, pse_np = generate_random_input(shape, pse_type)
    atten_mask_np = get_atten_mask(shape, pre_tokens, next_tokens, sparse_mode)

    ms.manual_seed(10)
    grads = speed_fusion_attention_backward_func(Tensor(query_np), Tensor(key_np), Tensor(value_np), head_num,
                                                 input_layout, pse=Tensor(pse_np), scale=scale, keep_prob=keep_prob,
                                                 atten_mask=Tensor(atten_mask_np), pre_tokens=pre_tokens,
                                                 next_tokens=next_tokens, sparse_mode=sparse_mode, pse_type=pse_type)

    dq = np.array([[[[-5.02586365e-04, -1.67131424e-04, -2.49385834e-04, 1.23620033e-04],
                     [6.51855469e-01, 2.35229492e-01, 3.82812500e-01, -9.74731445e-02],
                     [7.08007812e-01, 5.08789062e-01, 3.51074219e-01, 3.34960938e-01],
                     [3.38134766e-01, 2.29125977e-01, 1.26586914e-01, -9.07897949e-03]],
                    [[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                     [-3.10363770e-02, -6.62231445e-02, -4.51049805e-02, 9.04846191e-03],
                     [-4.68444824e-02, -6.45141602e-02, 3.48091125e-03, -1.89819336e-02],
                     [9.42382812e-02, 4.84313965e-02, 1.31225586e-01, 5.10559082e-02]]]]).astype(np.float16)
    dk = np.array([[[[-0.32592773, -0.39526367, -0.12768555, -0.17651367],
                     [-0.28002930, -0.24011230, -0.14062500, -0.25732422],
                     [0.58496094, 0.54443359, 0.31030273, 0.43115234],
                     [0.02061462, 0.09094238, -0.04229736, 0.00277138]],
                    [[0.04669189, 0.08477783, -0.04840088, 0.02378845],
                     [-0.03616333, -0.10516357, 0.11248779, -0.01355743],
                     [-0.02664185, 0.07073975, -0.27929688, -0.02407837],
                     [0.01611328, -0.05038452, 0.21508789, 0.01387024]]]]).astype(np.float16)
    dv = np.array([[[[2.22265625, 2.22265625, 2.22265625, 2.22265625],
                     [0.00000000, 0.00000000, 0.00000000, 0.00000000],
                     [1.04785156, 1.04785156, 1.04785156, 1.04785156],
                     [0.00000000, 0.00000000, 0.00000000, 0.00000000]],
                    [[1.01953125, 1.01953125, 1.01953125, 1.01953125],
                     [0.18811035, 0.18811035, 0.18811035, 0.18811035],
                     [0.58691406, 0.58691406, 0.58691406, 0.58691406],
                     [0.79394531, 0.79394531, 0.79394531, 0.79394531]]]]).astype(np.float16)
    np.testing.assert_allclose(grads[0].asnumpy(), dq, rtol=1e-3)
    np.testing.assert_allclose(grads[1].asnumpy(), dk, rtol=1e-3)
    np.testing.assert_allclose(grads[2].asnumpy(), dv, rtol=1e-3)

    atten_out = np.array([[[[2.63867188, 0.87744141, 1.30957031, -0.64892578],
                            [1.01464844, 0.33764648, 0.50341797, -0.24963379],
                            [-1.31054688, -1.17285156, -0.79394531, -1.11816406],
                            [0.32128906, 0.10681152, 0.15942383, -0.07904053]],
                           [[0.00000000, 0.00000000, 0.00000000, 0.00000000],
                            [0.21411133, 0.00695801, -0.29248047, 0.36376953],
                            [0.09552002, -0.01525116, 0.28100586, 0.19714355],
                            [0.36572266, -0.17395020, 1.66406250, 0.30957031]]]]).astype(np.float16)
    softmax_max = np.array(
        [[[[1.45018446, 1.45018446, 1.45018446, 1.45018446, 1.45018446, 1.45018446, 1.45018446, 1.45018446],
           [0.09358247, 0.09358247, 0.09358247, 0.09358247, 0.09358247, 0.09358247, 0.09358247, 0.09358247],
           [1.49721396, 1.49721396, 1.49721396, 1.49721396, 1.49721396, 1.49721396, 1.49721396, 1.49721396],
           [0.78424454, 0.78424454, 0.78424454, 0.78424454, 0.78424454, 0.78424454, 0.78424454, 0.78424454]],
          [[0.30972376, 0.30972376, 0.30972376, 0.30972376, 0.30972376, 0.30972376, 0.30972376, 0.30972376],
           [0.81006205, 0.81006205, 0.81006205, 0.81006205, 0.81006205, 0.81006205, 0.81006205, 0.81006205],
           [0.22907121, 0.22907121, 0.22907121, 0.22907121, 0.22907121, 0.22907121, 0.22907121, 0.22907121],
           [1.46355951, 1.46355951, 1.46355951, 1.46355951, 1.46355951, 1.46355951, 1.46355951, 1.46355951]]]]
        ).astype(np.float32)
    softmax_sum = np.array(
        [[[[1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000],
           [1.62510896, 1.62510896, 1.62510896, 1.62510896, 1.62510896, 1.62510896, 1.62510896, 1.62510896],
           [1.36333108, 1.36333108, 1.36333108, 1.36333108, 1.36333108, 1.36333108, 1.36333108, 1.36333108],
           [2.40727997, 2.40727997, 2.40727997, 2.40727997, 2.40727997, 2.40727997, 2.40727997, 2.40727997]],
          [[1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000],
           [1.54749846, 1.54749846, 1.54749846, 1.54749846, 1.54749846, 1.54749846, 1.54749846, 1.54749846],
           [2.43408227, 2.43408227, 2.43408227, 2.43408227, 2.43408227, 2.43408227, 2.43408227, 2.43408227],
           [1.79947615, 1.79947615, 1.79947615, 1.79947615, 1.79947615, 1.79947615, 1.79947615, 1.79947615]]]]
        ).astype(np.float32)

    ms.manual_seed(10)
    out = speed_fusion_attention_forward_func(Tensor(query_np), Tensor(key_np), Tensor(value_np), head_num,
                                              input_layout, pse=Tensor(pse_np), scale=scale, keep_prob=keep_prob,
                                              atten_mask=Tensor(atten_mask_np), pre_tokens=pre_tokens,
                                              next_tokens=next_tokens, sparse_mode=sparse_mode, pse_type=pse_type)
    np.testing.assert_allclose(out[0].asnumpy(), atten_out, rtol=1e-3)
    np.testing.assert_allclose(out[1].asnumpy(), softmax_max, rtol=1e-4)
    np.testing.assert_allclose(out[2].asnumpy(), softmax_sum, rtol=1e-4)
    assert out[3].shape == (0,)
    assert out[4] == Tensor(10, dtype=ms.int64)
    assert out[5] == Tensor(0, dtype=ms.int64)
    assert out[6] == Tensor(32, dtype=ms.int64)


def speed_fusion_attention_case3():
    """
    Feature: Test the precision for BNSD with pse_type is 3 and sparse_mode is 3.
    Description: Test function speed_fusion_attention forward and backward.
    Expectation: The result is correct.
    """
    shape = (1, 2, 4, 4)
    head_num = 2
    input_layout = "BNSD"
    sparse_mode = 3
    pre_tokens = 8
    next_tokens = 0
    scale = 0.4
    keep_prob = 0.9
    pse_type = 3

    query_np, key_np, value_np, pse_np = generate_random_input(shape, pse_type)
    atten_mask_np = get_atten_mask(shape, pre_tokens, next_tokens, sparse_mode)

    ms.manual_seed(10)
    grads = speed_fusion_attention_backward_func(Tensor(query_np), Tensor(key_np), Tensor(value_np), head_num,
                                                 input_layout, pse=Tensor(pse_np), scale=scale, keep_prob=keep_prob,
                                                 atten_mask=Tensor(atten_mask_np), pre_tokens=pre_tokens,
                                                 next_tokens=next_tokens, sparse_mode=sparse_mode, pse_type=pse_type)

    dq = np.array([[[[-8.41617584e-04, -2.79903412e-04, -4.17709351e-04, 2.07066536e-04],
                     [8.82812500e-01, 3.18603516e-01, 5.18554688e-01, -1.32080078e-01],
                     [4.85107422e-01, 3.72558594e-01, 2.34252930e-01, 2.70996094e-01],
                     [2.80761719e-01, 3.19580078e-01, -2.17590332e-02, -8.77075195e-02]],
                    [[-7.15255737e-07, -0.00000000e+00, 9.53674316e-07, -1.19209290e-06],
                     [-3.00598145e-02, -6.41479492e-02, -4.37316895e-02, 8.80432129e-03],
                     [-4.90417480e-02, -6.62841797e-02, 6.17218018e-03, -2.09960938e-02],
                     [6.10351562e-02, 2.96478271e-02, 1.00341797e-01, 3.07922363e-02]]]]).astype(np.float16)
    dk = np.array([[[[-0.28491211, -0.35888672, -0.11297607, -0.12902832],
                     [-0.21276855, -0.24401855, -0.04742432, -0.21289062],
                     [0.43579102, 0.33496094, 0.28369141, 0.33349609],
                     [0.06054688, 0.26708984, -0.12414551, 0.00813293]],
                    [[0.04635620, 0.07910156, -0.03408813, 0.02400208],
                     [-0.03860474, -0.08953857, 0.05410767, -0.01609802],
                     [-0.01916504, 0.04629517, -0.17297363, -0.01773071],
                     [0.01145172, -0.03579712, 0.15283203, 0.00985718]]]]).astype(np.float16)
    dv = np.array([[[[1.64453125, 1.64453125, 1.64453125, 1.64453125],
                     [0.71875000, 0.71875000, 0.71875000, 0.71875000],
                     [0.90283203, 0.90283203, 0.90283203, 0.90283203],
                     [0.50683594, 0.50683594, 0.50683594, 0.50683594]],
                    [[1.79687500, 1.79687500, 1.79687500, 1.79687500],
                     [0.10693359, 0.10693359, 0.10693359, 0.10693359],
                     [0.69726562, 0.69726562, 0.69726562, 0.69726562],
                     [0.72558594, 0.72558594, 0.72558594, 0.72558594]]]]).astype(np.float16)
    np.testing.assert_allclose(grads[0].asnumpy(), dq, rtol=1e-3)
    np.testing.assert_allclose(grads[1].asnumpy(), dk, rtol=1e-3)
    np.testing.assert_allclose(grads[2].asnumpy(), dv, rtol=1e-3)

    atten_out = np.array([[[[2.05273438, 0.68261719, 1.01855469, -0.50488281],
                            [0.47167969, 0.11206055, 0.09014893, -0.26855469],
                            [-1.19335938, -1.03125000, -0.71582031, -0.94824219],
                            [0.04498291, -0.65771484, 0.44262695, -0.07385254]],
                           [[0.47070312, 0.01529694, -0.64306641, 0.79980469],
                            [0.14575195, 0.00473404, -0.19909668, 0.24755859],
                            [0.06167603, -0.01316833, 0.25585938, 0.13354492],
                            [0.22436523, -0.24414062, 1.71972656, 0.17968750]]]]).astype(np.float16)
    softmax_max = np.array(
        [[[[1.93357933, 1.93357933, 1.93357933, 1.93357933, 1.93357933, 1.93357933, 1.93357933, 1.93357933],
           [0.12477662, 0.12477662, 0.12477662, 0.12477662, 0.12477662, 0.12477662, 0.12477662, 0.12477662],
           [1.99628532, 1.99628532, 1.99628532, 1.99628532, 1.99628532, 1.99628532, 1.99628532, 1.99628532],
           [1.04565942, 1.04565942, 1.04565942, 1.04565942, 1.04565942, 1.04565942, 1.04565942, 1.04565942]],
          [[0.41296503, 0.41296503, 0.41296503, 0.41296503, 0.41296503, 0.41296503, 0.41296503, 0.41296503],
           [1.08008277, 1.08008277, 1.08008277, 1.08008277, 1.08008277, 1.08008277, 1.08008277, 1.08008277],
           [0.30542827, 0.30542827, 0.30542827, 0.30542827, 0.30542827, 0.30542827, 0.30542827, 0.30542827],
           [1.95141256, 1.95141256, 1.95141256, 1.95141256, 1.95141256, 1.95141256, 1.95141256, 1.95141256]]]]
        ).astype(np.float32)
    softmax_sum = np.array(
        [[[[1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000],
           [1.54574370, 1.54574370, 1.54574370, 1.54574370, 1.54574370, 1.54574370, 1.54574370, 1.54574370],
           [1.23072267, 1.23072267, 1.23072267, 1.23072267, 1.23072267, 1.23072267, 1.23072267, 1.23072267],
           [2.19226146, 2.19226146, 2.19226146, 2.19226146, 2.19226146, 2.19226146, 2.19226146, 2.19226146]],
          [[1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000],
           [1.44847870, 1.44847870, 1.44847870, 1.44847870, 1.44847870, 1.44847870, 1.44847870, 1.44847870],
           [2.28967285, 2.28967285, 2.28967285, 2.28967285, 2.28967285, 2.28967285, 2.28967285, 2.28967285],
           [1.53119659, 1.53119659, 1.53119659, 1.53119659, 1.53119659, 1.53119659, 1.53119659, 1.53119659]]]]
        ).astype(np.float32)

    ms.manual_seed(10)
    out = speed_fusion_attention_forward_func(Tensor(query_np), Tensor(key_np), Tensor(value_np), head_num,
                                              input_layout, pse=Tensor(pse_np), scale=scale, keep_prob=keep_prob,
                                              atten_mask=Tensor(atten_mask_np), pre_tokens=pre_tokens,
                                              next_tokens=next_tokens, sparse_mode=sparse_mode, pse_type=pse_type)
    np.testing.assert_allclose(out[0].asnumpy(), atten_out, rtol=1e-3)
    np.testing.assert_allclose(out[1].asnumpy(), softmax_max, rtol=1e-4)
    np.testing.assert_allclose(out[2].asnumpy(), softmax_sum, rtol=1e-4)
    assert out[3].shape == (0,)
    assert out[4] == Tensor(10, dtype=ms.int64)
    assert out[5] == Tensor(0, dtype=ms.int64)
    assert out[6] == Tensor(32, dtype=ms.int64)


def speed_fusion_attention_case4():
    """
    Feature: Test the precision for BNSD with pse_type is 3 and sparse_mode is 4.
    Description: Test function speed_fusion_attention forward and backward.
    Expectation: The result is correct.
    """
    shape = (1, 2, 4, 4)
    head_num = 2
    input_layout = "BNSD"
    sparse_mode = 4
    pre_tokens = 2
    next_tokens = 2
    scale = 0.5
    keep_prob = 1.0
    pse_type = 3

    query_np, key_np, value_np, pse_np = generate_random_input(shape, pse_type)

    grads = speed_fusion_attention_backward_func(Tensor(query_np), Tensor(key_np), Tensor(value_np), head_num,
                                                 input_layout, pse=Tensor(pse_np), scale=scale, keep_prob=keep_prob,
                                                 atten_mask=None, pre_tokens=pre_tokens, next_tokens=next_tokens,
                                                 sparse_mode=sparse_mode, pse_type=pse_type)

    dq = np.array([[[[0.49267578, 0.31079102, 0.19604492, -0.03540039],
                     [1.16992188, 0.62255859, 0.68994141, 0.35034180],
                     [0.40087891, 0.13977051, 0.34790039, 0.29345703],
                     [0.80175781, 0.33374023, 0.56396484, 0.29785156]],
                    [[0.15905762, 0.25561523, 0.14624023, 0.01330566],
                     [0.18554688, 0.28979492, 0.03869629, 0.04693604],
                     [0.11383057, 0.10272217, 0.11828613, 0.04751587],
                     [0.05209351, 0.01346588, 0.09991455, 0.02937317]]]]).astype(np.float16)
    dk = np.array([[[[0.16845703, -0.23095703, 0.17211914, -0.17211914],
                     [-0.37500000, -0.28417969, -0.18676758, -0.11450195],
                     [0.65576172, 0.97851562, 0.15136719, 0.42822266],
                     [-0.44921875, -0.46313477, -0.13659668, -0.14172363]],
                    [[-0.20739746, -0.15429688, -0.02813721, -0.20104980],
                     [0.32446289, 0.26440430, 0.10656738, 0.28930664],
                     [-0.12634277, -0.06414795, -0.28540039, -0.10491943],
                     [0.00929260, -0.04586792, 0.20727539, 0.01663208]]]]).astype(np.float16)
    dv = np.array([[[[1.11328125, 1.11328125, 1.11328125, 1.11328125],
                     [0.54882812, 0.54882812, 0.54882812, 0.54882812],
                     [1.43359375, 1.43359375, 1.43359375, 1.43359375],
                     [0.90429688, 0.90429688, 0.90429688, 0.90429688]],
                    [[0.74121094, 0.74121094, 0.74121094, 0.74121094],
                     [1.09179688, 1.09179688, 1.09179688, 1.09179688],
                     [0.71679688, 0.71679688, 0.71679688, 0.71679688],
                     [1.45019531, 1.45019531, 1.45019531, 1.45019531]]]]).astype(np.float16)
    np.testing.assert_allclose(grads[0].asnumpy(), dq, rtol=1e-3)
    np.testing.assert_allclose(grads[1].asnumpy(), dk, rtol=1e-3)
    np.testing.assert_allclose(grads[2].asnumpy(), dv, rtol=1e-3)

    atten_out = np.array([[[[1.60644531, 0.46069336, 0.82568359, -0.42773438],
                            [-0.45385742, -0.73193359, -0.14514160, -0.49487305],
                            [-1.10156250, -1.07910156, -0.56250000, -0.80664062],
                            [-0.45629883, -1.02734375, 0.12396240, -0.37084961]],
                           [[0.47729492, 0.31445312, 0.27221680, 0.45141602],
                            [0.60351562, 0.55029297, 0.42407227, 0.43774414],
                            [0.26367188, -0.02201843, 1.12304688, 0.21533203],
                            [0.18151855, -0.29492188, 1.70605469, 0.14526367]]]]).astype(np.float16)
    softmax_max = np.array(
        [[[[2.41697407, 2.41697407, 2.41697407, 2.41697407, 2.41697407, 2.41697407, 2.41697407, 2.41697407],
           [0.50237018, 0.50237018, 0.50237018, 0.50237018, 0.50237018, 0.50237018, 0.50237018, 0.50237018],
           [2.49535656, 2.49535656, 2.49535656, 2.49535656, 2.49535656, 2.49535656, 2.49535656, 2.49535656],
           [1.30707419, 1.30707419, 1.30707419, 1.30707419, 1.30707419, 1.30707419, 1.30707419, 1.30707419]],
          [[0.51620626, 0.51620626, 0.51620626, 0.51620626, 0.51620626, 0.51620626, 0.51620626, 0.51620626],
           [1.35010338, 1.35010338, 1.35010338, 1.35010338, 1.35010338, 1.35010338, 1.35010338, 1.35010338],
           [0.90221488, 0.90221488, 0.90221488, 0.90221488, 0.90221488, 0.90221488, 0.90221488, 0.90221488],
           [2.43926573, 2.43926573, 2.43926573, 2.43926573, 2.43926573, 2.43926573, 2.43926573, 2.43926573]]]]
        ).astype(np.float32)
    softmax_sum = np.array(
        [[[[1.11589324, 1.11589324, 1.11589324, 1.11589324, 1.11589324, 1.11589324, 1.11589324, 1.11589324],
           [2.63446760, 2.63446760, 2.63446760, 2.63446760, 2.63446760, 2.63446760, 2.63446760, 2.63446760],
           [1.32268143, 1.32268143, 1.32268143, 1.32268143, 1.32268143, 1.32268143, 1.32268143, 1.32268143],
           [1.99203920, 1.99203920, 1.99203920, 1.99203920, 1.99203920, 1.99203920, 1.99203920, 1.99203920]],
          [[2.62266779, 2.62266779, 2.62266779, 2.62266779, 2.62266779, 2.62266779, 2.62266779, 2.62266779],
           [1.89210248, 1.89210248, 1.89210248, 1.89210248, 1.89210248, 1.89210248, 1.89210248, 1.89210248],
           [2.28339052, 2.28339052, 2.28339052, 2.28339052, 2.28339052, 2.28339052, 2.28339052, 2.28339052],
           [1.35696399, 1.35696399, 1.35696399, 1.35696399, 1.35696399, 1.35696399, 1.35696399, 1.35696399]]]]
        ).astype(np.float32)

    out = speed_fusion_attention_forward_func(Tensor(query_np), Tensor(key_np), Tensor(value_np), head_num,
                                              input_layout, pse=Tensor(pse_np), scale=scale, keep_prob=keep_prob,
                                              atten_mask=None, pre_tokens=pre_tokens, next_tokens=next_tokens,
                                              sparse_mode=sparse_mode, pse_type=pse_type)
    np.testing.assert_allclose(out[0].asnumpy(), atten_out, rtol=1e-3)
    np.testing.assert_allclose(out[1].asnumpy(), softmax_max, rtol=1e-4)
    np.testing.assert_allclose(out[2].asnumpy(), softmax_sum, rtol=1e-4)
    assert out[3].shape == (0,)
    assert out[6] == Tensor(32, dtype=ms.int64)


@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_speed_fusion_attention_normal(mode):
    """
    Feature: Test the precision for ops.speed_fusion_attention.
    Description: Test function forward and backward.
    Expectation: The result is correct.
    """
    ms.context.set_context(mode=mode)

    #speed_fusion_attention_case0()
    #speed_fusion_attention_case1()
    speed_fusion_attention_case2()
    speed_fusion_attention_case3()
    speed_fusion_attention_case4()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=34.0)
@pytest.mark.parametrize('input_layout', ["BSH", "SBH", "BNSD", "BSND", "TND"])
def test_speed_fusion_attention_dynamic(input_layout):
    """
    Feature: Test the precision for ops.speed_fusion_attention with dynamic.
    Description: Test function forward and backward.
    Expectation: The result is correct.
    """
    def speed_fusion_attention_func(query, key, value, head_num, input_layout, actual_seq_qlen, actual_seq_kvlen):
        return speed_fusion_attention(query, key, value, head_num, input_layout, actual_seq_qlen=actual_seq_qlen,
                                      actual_seq_kvlen=actual_seq_kvlen)

    input1_shape = {
        "BSH": (1, 8, 512),
        "SBH": (8, 1, 512),
        "BNSD": (1, 4, 8, 128),
        "BSND": (1, 8, 4, 128),
        "TND": (8, 4, 128),
    }
    input2_shape = {
        "BSH": (4, 2, 1024),
        "SBH": (2, 4, 1024),
        "BNSD": (4, 8, 2, 128),
        "BSND": (4, 2, 8, 128),
        "TND": (8, 8, 128),
    }

    head_num1 = 4
    head_num2 = 8
    query_np1, key_np1, value_np1, _ = generate_random_input(input1_shape[input_layout], 1)
    query_np2, key_np2, value_np2, _ = generate_random_input(input2_shape[input_layout], 1)
    actual_seq_qlen = (2, 8)
    actual_seq_kvlen = (2, 8)
    if input_layout != "TND":
        actual_seq_qlen = None
        actual_seq_kvlen = None

    input_seq1 = [Tensor(query_np1), Tensor(key_np1), Tensor(value_np1), head_num1, input_layout, actual_seq_qlen,
                  actual_seq_kvlen]
    input_seq2 = [Tensor(query_np2), Tensor(key_np2), Tensor(value_np2), head_num2, input_layout, actual_seq_qlen,
                  actual_seq_kvlen]

    if input_layout in ["BNSD", "BSND"]:
        disable_case = ['EmptyTensor', 'ScalarTensor']
    else:
        disable_case = ['ScalarTensor']

    TEST_OP(speed_fusion_attention_func, [input_seq1, input_seq2],
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'],
            disable_case=disable_case,
            case_config={'disable_input_check': True,
                         'all_dim_zero': True,
                         "ignore_output_index": 5})


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=34.0)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_speed_fusion_attention_rng_state(mode):
    """
    Feature: Test the precision for ops.speed_fusion_attention with random state.
    Description: Test function forward.
    Expectation: The result is correct.
    """
    ms.context.set_context(mode=mode)

    shape = (1, 8, 256, 256)
    head_num = 8
    input_layout = "BNSD"
    sparse_mode = 0
    pre_tokens = 128
    next_tokens = 128
    scale = 0.5
    keep_prob = 0.3
    pse_type = 0

    query_np, key_np, value_np, pse_np = generate_random_input(shape, pse_type)

    state = ms.get_rng_state()
    out0 = speed_fusion_attention_forward_func(Tensor(query_np), Tensor(key_np), Tensor(value_np), head_num,
                                               input_layout, pse=Tensor(pse_np), scale=scale, keep_prob=keep_prob,
                                               atten_mask=None, pre_tokens=pre_tokens, next_tokens=next_tokens,
                                               sparse_mode=sparse_mode, pse_type=pse_type)

    out1 = speed_fusion_attention_forward_func(Tensor(query_np), Tensor(key_np), Tensor(value_np), head_num,
                                               input_layout, pse=Tensor(pse_np), scale=scale, keep_prob=keep_prob,
                                               atten_mask=None, pre_tokens=pre_tokens, next_tokens=next_tokens,
                                               sparse_mode=sparse_mode, pse_type=pse_type)
    ms.set_rng_state(state)
    out2 = speed_fusion_attention_forward_func(Tensor(query_np), Tensor(key_np), Tensor(value_np), head_num,
                                               input_layout, pse=Tensor(pse_np), scale=scale, keep_prob=keep_prob,
                                               atten_mask=None, pre_tokens=pre_tokens, next_tokens=next_tokens,
                                               sparse_mode=sparse_mode, pse_type=pse_type)

    assert not (out0[0].asnumpy() == out1[0].asnumpy()).all()
    assert (out0[0].asnumpy() == out2[0].asnumpy()).all()

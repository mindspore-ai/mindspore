# Copyright 2024 Huawei Technologies Co., Ltd
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

"""test embedding"""

import numpy as np

import mindspore as ms
from mindspore import ops, nn
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark


def embedding_func(input_x, weight, padding_idx=None, max_norm=None, norm_type=2.0):
    """embedding_func"""
    return ops.embedding(input_x, weight, padding_idx, max_norm=max_norm, norm_type=norm_type)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_embedding_static_shape():
    """
    Feature: static shape of embedding.
    Description: static shape of embedding.
    Expectation: expect correct result.
    """
    input_x = ms.Tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    weight = ms.Parameter([[0.3649, 0.6303, 0.7726],
                           [0.4307, 0.7575, 0.7544],
                           [0.5400, 0.6909, 0.9423],
                           [0.7787, 0.0619, 0.6290],
                           [0.3424, 0.4064, 0.9990],
                           [0.5328, 0.5363, 0.3558],
                           [0.5438, 0.9858, 0.8243],
                           [0.0440, 0.2209, 0.9105],
                           [0.1723, 0.6084, 0.4130],
                           [0.6587, 0.5972, 0.7808]])

    expect1 = [[[0.4307, 0.7575, 0.7544],
                [0.5400, 0.6909, 0.9423],
                [0.3424, 0.4064, 0.9990],
                [0.5328, 0.5363, 0.3558]],
               [[0.3424, 0.4064, 0.9990],
                [0.7787, 0.0619, 0.6290],
                [0.5400, 0.6909, 0.9423],
                [0.6587, 0.5972, 0.7808]]]

    ms_out1 = ops.embedding(input_x, weight, padding_idx=None)
    assert np.allclose(ms_out1.asnumpy(), expect1, rtol=1e-4)

    expect2 = [[[0.0087, 0.0154, 0.0153],
                [0.0097, 0.0125, 0.0170],
                [0.0082, 0.0097, 0.0239],
                [0.0146, 0.0147, 0.0097]],
               [[0.0082, 0.0097, 0.0239],
                [0.0268, 0.0021, 0.0216],
                [0.0097, 0.0125, 0.0170],
                [0.0125, 0.0113, 0.0148]]]

    expect_w = [[0.3649, 0.6303, 0.7726],
                [0.0087, 0.0154, 0.0153],
                [0.0097, 0.0125, 0.0170],
                [0.0268, 0.0021, 0.0216],
                [0.0082, 0.0097, 0.0239],
                [0.0146, 0.0147, 0.0097],
                [0.5438, 0.9858, 0.8243],
                [0.0440, 0.2209, 0.9105],
                [0.1723, 0.6084, 0.4130],
                [0.0125, 0.0113, 0.0148]]

    ms_out2 = ops.embedding(input_x, weight, max_norm=0.5, norm_type=0.3)

    assert np.allclose(ms_out2.asnumpy(), expect2, rtol=1e-4, atol=1e-4)
    assert np.allclose(weight.asnumpy(), expect_w, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_embedding_dynamic_shape():
    """
    Feature: dynamic shape of embedding.
    Description: dynamic shape of embedding.
    Expectation: expect correct result.
    """
    input1 = ms.Tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    input2 = ms.Tensor(np.random.randint(0, 10, size=(18, 17, 19, 14)))
    weight = ms.Parameter(np.random.rand(10, 3).astype(np.float32))

    TEST_OP(embedding_func, [[input1, weight, 0, 0.3, 1.1], [input2, weight, -1, 0.6, 2.4]], '',
            disable_input_check=True, disable_yaml_check=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_embedding_grad():
    """
    Feature: grad of embedding.
    Description: test grad using different padding_idx, max_norm and scale_grad_by_freq
    Expectation: expect correct result.
    """
    def embedding_func1(input_x, weight, padding_idx, max_norm, norm_type, scale_grad):
        return ops.embedding(input_x, weight, padding_idx, max_norm, norm_type, scale_grad)

    def grad_func1(input_x, weight, padding_idx, max_norm, norm_type, scale_grad):
        return ops.grad(embedding_func1, grad_position=1)(input_x, weight, padding_idx, max_norm, norm_type, scale_grad)

    def embedding_func2(input_x, weight, padding_idx, max_norm, norm_type, scale_grad):
        return ops.embedding(input_x, weight, padding_idx, max_norm, norm_type, scale_grad)

    def grad_func2(input_x, weight, padding_idx, max_norm, norm_type, scale_grad):
        return ops.grad(embedding_func2, grad_position=1)(input_x, weight, padding_idx, max_norm, norm_type, scale_grad)

    input_x = ms.Tensor([[0, 2, 4, 5], [4, 3, 2, 9]])
    weight = ms.Parameter([[0.3649, 0.6303, 0.7726],
                           [0.4307, 0.7575, 0.7544],
                           [0.5400, 0.6909, 0.9423],
                           [0.7787, 0.0619, 0.6290],
                           [0.3424, 0.4064, 0.9990],
                           [0.5328, 0.5363, 0.3558],
                           [0.5438, 0.9858, 0.8243],
                           [0.0440, 0.2209, 0.9105],
                           [0.1723, 0.6084, 0.4130],
                           [0.6587, 0.5972, 0.7808]])

    expect_out = [[1., 1., 1.],
                  [0., 0., 0.],
                  [0., 0., 0.],
                  [1., 1., 1.],
                  [2., 2., 2.],
                  [1., 1., 1.],
                  [0., 0., 0.],
                  [0., 0., 0.],
                  [0., 0., 0.],
                  [1., 1., 1.]]

    ms_out = grad_func1(input_x, weight, 2, 0.3, 1.6, False)
    assert np.allclose(ms_out.asnumpy(), expect_out, rtol=1e-4, atol=1e-4)

    ms_out = grad_func2(input_x, weight, -8, 0.3, 1.6, False)
    assert np.allclose(ms_out.asnumpy(), expect_out, rtol=1e-4, atol=1e-4)

    def embedding_func3(input_x, weight, max_norm, norm_type, scale_grad):
        return ops.embedding(input_x, weight, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad)

    def grad_func3(input_x, weight, max_norm, norm_type, scale_grad):
        return ops.grad(embedding_func3, grad_position=1)(input_x, weight, max_norm, norm_type, scale_grad)

    expect_out3 = [[1., 1., 1.],
                   [0., 0., 0.],
                   [2., 2., 2.],
                   [1., 1., 1.],
                   [2., 2., 2.],
                   [1., 1., 1.],
                   [0., 0., 0.],
                   [0., 0., 0.],
                   [0., 0., 0.],
                   [1., 1., 1.]]

    ms_out = grad_func3(input_x, weight, 0.3, 1.6, False)
    assert np.allclose(ms_out.asnumpy(), expect_out3, rtol=1e-4, atol=1e-4)

    def embedding_func4(input_x, weight, max_norm, norm_type, scale_grad):
        return ops.embedding(input_x, weight, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad)

    def grad_func4(input_x, weight, max_norm, norm_type, scale_grad):
        return ops.grad(embedding_func4, grad_position=1)(input_x, weight, max_norm, norm_type, scale_grad)

    expect_out4 = [[1., 1., 1.],
                   [0., 0., 0.],
                   [1., 1., 1.],
                   [1., 1., 1.],
                   [1., 1., 1.],
                   [1., 1., 1.],
                   [0., 0., 0.],
                   [0., 0., 0.],
                   [0., 0., 0.],
                   [1., 1., 1.]]
    ms_out = grad_func4(input_x, weight, 0.3, 1.6, True)
    assert np.allclose(ms_out.asnumpy(), expect_out4, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_embedding_nn_api():
    """
    Feature: nn.extend.Embedding.
    Description: nn.extend.Embedding.
    Expectation: expect correct result.
    """
    input_x = ms.Tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    weight = ms.Tensor([[0.3649, 0.6303, 0.7726],
                        [0.4307, 0.7575, 0.7544],
                        [0.5400, 0.6909, 0.9423],
                        [0.7787, 0.0619, 0.6290],
                        [0.3424, 0.4064, 0.9990],
                        [0.5328, 0.5363, 0.3558],
                        [0.5438, 0.9858, 0.8243],
                        [0.0440, 0.2209, 0.9105],
                        [0.1723, 0.6084, 0.4130],
                        [0.6587, 0.5972, 0.7808]])

    expect = [[[0.1495, 0.2629, 0.2618],
               [0.1678, 0.2147, 0.2928],
               [0.1210, 0.1437, 0.3531],
               [0.2551, 0.2568, 0.1703]],
              [[0.1210, 0.1437, 0.3531],
               [0.3106, 0.0247, 0.2509],
               [0.1678, 0.2147, 0.2928],
               [0.2227, 0.2019, 0.2639]]]

    # exception cases
    # embedding = nn.extend.Embedding(num_embeddings=4.0, embedding_dim=weight.shape[1])
    # embedding = nn.extend.Embedding(num_embeddings=weight.shape[0], embedding_dim=3.0)
    # embedding = nn.extend.Embedding(num_embeddings=weight.shape[0], embedding_dim=weight.shape[1], dtype='str')
    # embedding = nn.extend.Embedding(num_embeddings=weight.shape[0], embedding_dim=weight.shape[1], max_norm=[1])
    # embedding = nn.extend.Embedding(num_embeddings=weight.shape[0], embedding_dim=weight.shape[1], max_norm=0.4, norm_type=False)
    # embedding = nn.extend.Embedding(num_embeddings=weight.shape[0], embedding_dim=weight.shape[1], padding_idx=18, max_norm=0.4, norm_type=2.0)
    # embedding = nn.extend.Embedding(num_embeddings=weight.shape[0], embedding_dim=weight.shape[1], padding_idx=8, max_norm=0.4, norm_type=2.0, scale_grad_by_freq=1.0)
    embedding_layer = nn.extend.Embedding(num_embeddings=weight.shape[0], embedding_dim=weight.shape[1],
                                          padding_idx=8, max_norm=0.4, norm_type=2.0, scale_grad_by_freq=True,
                                          _weight=weight)
    ms_out1 = embedding_layer(input_x)
    assert np.allclose(ms_out1.asnumpy(), expect, rtol=1e-3)

    expect_w = [[0.3649, 0.6303, 0.7726],
                [0.1495, 0.2629, 0.2618],
                [0.1678, 0.2147, 0.2928],
                [0.3106, 0.0247, 0.2509],
                [0.1210, 0.1437, 0.3531],
                [0.2551, 0.2568, 0.1703],
                [0.5438, 0.9858, 0.8243],
                [0.0440, 0.2209, 0.9105],
                [0.1723, 0.6084, 0.4130],
                [0.2227, 0.2019, 0.2639]]
    assert np.allclose(embedding_layer.weight.value().asnumpy(), expect_w, rtol=1e-3)

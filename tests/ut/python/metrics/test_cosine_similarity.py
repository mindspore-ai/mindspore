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
"""test cosine_similarity"""
import pytest
import numpy as np
from sklearn.metrics import pairwise
from mindspore.nn.metrics import CosineSimilarity


def test_cosine_similarity():
    """test_cosine_similarity"""
    test_data = np.array([[5, 8, 3, 2], [5, 8, 3, 2], [4, 2, 3, 4]])
    metric = CosineSimilarity()
    metric.clear()
    metric.update(test_data)
    square_matrix = metric.eval()

    assert np.allclose(square_matrix, np.array([[0, 1, 0.78229315], [1, 0, 0.78229315], [0.78229315, 0.78229315, 0]]))


def test_cosine_similarity_compare():
    """test_cosine_similarity_compare"""
    test_data = np.array([[5, 8, 3, 2], [5, 8, 3, 2], [4, 2, 3, 4]])
    metric = CosineSimilarity(similarity='cosine', reduction='none', zero_diagonal=False)
    metric.clear()
    metric.update(test_data)
    ms_square_matrix = metric.eval()

    def sklearn_cosine_similarity(test_data, similarity, reduction):
        """sklearn_cosine_similarity"""
        metric_func = {'cosine': pairwise.cosine_similarity,
                       'dot': pairwise.linear_kernel}[similarity]

        square_matrix = metric_func(test_data, test_data)
        if reduction == 'mean':
            return square_matrix.mean(axis=-1)
        if reduction == 'sum':
            return square_matrix.sum(axis=-1)
        return square_matrix

    sk_square_matrix = sklearn_cosine_similarity(test_data, similarity='cosine', reduction='none')

    assert np.allclose(sk_square_matrix, ms_square_matrix)


def test_cosine_similarity_init1():
    """test_cosine_similarity_init1"""
    with pytest.raises(ValueError):
        CosineSimilarity(similarity="4")


def test_cosine_similarity_init2():
    """test_cosine_similarity_init2"""
    with pytest.raises(TypeError):
        CosineSimilarity(similarity=4)


def test_cosine_similarity_init3():
    """test_cosine_similarity_init3"""
    with pytest.raises(TypeError):
        CosineSimilarity(reduction=2)


def test_cosine_similarity_init4():
    """test_cosine_similarity_init4"""
    with pytest.raises(ValueError):
        CosineSimilarity(reduction="1")



def test_cosine_similarity_init5():
    """test_cosine_similarity_init5"""
    with pytest.raises(TypeError):
        CosineSimilarity(zero_diagonal=3)


def test_cosine_similarity_runtime():
    """test_cosine_similarity_runtime"""
    metric = CosineSimilarity()
    metric.clear()

    with pytest.raises(RuntimeError):
        metric.eval()

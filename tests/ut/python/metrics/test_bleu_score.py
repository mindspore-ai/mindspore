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
"""test_bleu_score"""
import math
import pytest
from mindspore.nn.metrics import BleuScore


def test_bleu_score():
    """test_bleu_score"""
    candidate_corpus = [['i', 'have', 'a', 'pen', 'on', 'my', 'desk']]
    reference_corpus = [[['i', 'have', 'a', 'pen', 'in', 'my', 'desk'],
                         ['there', 'is', 'a', 'pen', 'on', 'the', 'desk']]]
    metric = BleuScore(n_gram=4, smooth=False)
    metric.clear()
    metric.update(candidate_corpus, reference_corpus)
    bleu_score = metric.eval()

    assert math.isclose(bleu_score, 0.5946035575013605, abs_tol=0.0001)


def test_bleu_score_update1():
    """test_bleu_score_update1"""
    candidate_corpus = ['the cat is on the mat'.split()]
    metric = BleuScore()
    metric.clear()

    with pytest.raises(ValueError):
        metric.update(candidate_corpus)


def test_bleu_score_update2():
    """test_bleu_score_update2"""
    candidate_corpus = [['the cat is on the mat'.split()], ['a cat is on the mat'.split()]]
    reference_corpus = [['there is a cat on the mat'.split(), 'a cat is on the mat'.split()]]
    metric = BleuScore()
    metric.clear()

    with pytest.raises(ValueError):
        metric.update(candidate_corpus, reference_corpus)


def test_bleu_score_init1():
    """test_bleu_score_init1"""
    with pytest.raises(TypeError):
        BleuScore(n_gram="3")


def test_bleu_score_init2():
    """test_bleu_score_init2"""
    with pytest.raises(TypeError):
        BleuScore(smooth=5)


def test_bleu_score_runtime():
    """test_bleu_score_runtime"""
    metric = BleuScore()
    metric.clear()

    with pytest.raises(RuntimeError):
        metric.eval()

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
""" test_embedding """
import numpy as np

from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.model_zoo.Bert_NEZHA import EmbeddingLookup, EmbeddingPostprocessor
from ..ut_filter import non_graph_engine


@non_graph_engine
def test_check_embedding_lookup_1():
    m = EmbeddingLookup(vocab_size=32000,
                        embedding_size=768,
                        embedding_shape=[1, 128, 768],
                        use_one_hot_embeddings=False)
    m(Tensor(np.ones([128]), mstype.int32))


@non_graph_engine
def test_check_embedding_lookup_2():
    m = EmbeddingLookup(vocab_size=32000,
                        embedding_size=768,
                        embedding_shape=[1, 128, 768],
                        use_one_hot_embeddings=True)
    m(Tensor(np.ones([128]), mstype.int32))


@non_graph_engine
def test_check_embedding_lookup_3():
    m = EmbeddingLookup(vocab_size=32000,
                        embedding_size=768,
                        embedding_shape=[1, 128, 768],
                        use_one_hot_embeddings=True,
                        initializer_range=0.01)
    m(Tensor(np.ones([128]), mstype.int32))


@non_graph_engine
def test_embedding_post_1():
    m = EmbeddingPostprocessor(embedding_size=768,
                               embedding_shape=[1, 128, 768],
                               use_token_type=True)
    m(Tensor(np.ones([128]), mstype.int32), Tensor(np.ones([1, 128, 768]), mstype.float32))


@non_graph_engine
def test_embedding_post_2():
    m = EmbeddingPostprocessor(embedding_size=768,
                               embedding_shape=[1, 128, 768],
                               use_token_type=True,
                               initializer_range=0.3)
    m(Tensor(np.ones([128]), mstype.int32), Tensor(np.ones([1, 128, 768]), mstype.float32))

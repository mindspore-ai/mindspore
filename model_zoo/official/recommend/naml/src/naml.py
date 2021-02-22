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
"""NAML network."""
import mindspore.nn as nn
from mindspore.common import initializer as init
import mindspore.ops as ops

class Attention(nn.Cell):
    """
    Softmax attention implement.

    Args:
        query_vector_dim (int): dimension of the query vector in attention.
        input_vector_dim (int): dimension of the input vector in attention.

    Input:
        input (Tensor): input tensor, shape is (batch_size, n_input_vector, input_vector_dim)

    Returns:
        Tensor, output tensor, shape is (batch_size, n_input_vector).

    Examples:
        >>> Attention(query_vector_dim, input_vector_dim)
    """
    def __init__(self, query_vector_dim, input_vector_dim):
        super(Attention, self).__init__()
        self.dense1 = nn.Dense(input_vector_dim, query_vector_dim, has_bias=True, activation='tanh')
        self.dense2 = nn.Dense(query_vector_dim, 1, has_bias=False)
        self.softmax = nn.Softmax()
        self.sum_keep_dims = ops.ReduceSum(keep_dims=True)
        self.sum = ops.ReduceSum(keep_dims=False)

    def construct(self, x):
        dtype = ops.dtype(x)
        batch_size, n_input_vector, input_vector_dim = ops.shape(x)
        feature = ops.reshape(x, (-1, input_vector_dim))
        attention = ops.reshape(self.dense2(self.dense1(feature)), (batch_size, n_input_vector))
        attention_weight = ops.cast(self.softmax(attention), dtype)
        weighted_input = x * ops.expand_dims(attention_weight, 2)
        return self.sum(weighted_input, 1)

class NewsEncoder(nn.Cell):
    """
    The main function to create news encoder of NAML.

    Args:
        args (class): global hyper-parameters.
        word_embedding (Tensor): parameter of word embedding.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> NewsEncoder(args, embedding_table)
    """
    def __init__(self, args, embedding_table=None):
        super(NewsEncoder, self).__init__()
        # categories
        self.category_embedding = nn.Embedding(args.n_categories, args.category_embedding_dim)
        self.category_dense = nn.Dense(args.category_embedding_dim, args.n_filters, has_bias=True, activation="relu")

        self.sub_category_embedding = nn.Embedding(args.n_sub_categories, args.category_embedding_dim)
        self.subcategory_dense = nn.Dense(args.category_embedding_dim, args.n_filters, has_bias=True, activation="relu")

        # title and abstract
        if embedding_table is None:
            word_embedding = [nn.Embedding(args.n_words, args.word_embedding_dim)]
        else:
            word_embedding = [nn.Embedding(args.n_words, args.word_embedding_dim, embedding_table=embedding_table)]
        title_CNN = [
            nn.Conv1d(args.word_embedding_dim, args.n_filters, kernel_size=args.window_size, pad_mode='same',
                      has_bias=True),
            nn.ReLU()
        ]
        abstract_CNN = [
            nn.Conv1d(args.word_embedding_dim, args.n_filters, kernel_size=args.window_size, pad_mode='same',
                      has_bias=True),
            nn.ReLU()
        ]
        if args.phase == "train":
            word_embedding.append(nn.Dropout(keep_prob=(1-args.dropout_ratio)))
            title_CNN.append(nn.Dropout(keep_prob=(1-args.dropout_ratio)))
            abstract_CNN.append(nn.Dropout(keep_prob=(1-args.dropout_ratio)))
        self.word_embedding = nn.SequentialCell(word_embedding)
        self.title_CNN = nn.SequentialCell(title_CNN)
        self.abstract_CNN = nn.SequentialCell(abstract_CNN)
        self.title_attention = Attention(args.query_vector_dim, args.n_filters)
        self.abstract_attention = Attention(args.query_vector_dim, args.n_filters)
        self.total_attention = Attention(args.query_vector_dim, args.n_filters)
        self.pack = ops.Stack(axis=1)
        self.title_shape = (-1, args.n_words_title)
        self.abstract_shape = (-1, args.n_words_abstract)

    def construct(self, category, subcategory, title, abstract):
        """
        The news encoder is composed of title encoder, abstract encoder, category encoder and subcategory encoder.
        """
        # Categories
        category_embedded = self.category_embedding(ops.reshape(category, (-1,)))
        category_vector = self.category_dense(category_embedded)
        subcategory_embedded = self.sub_category_embedding(ops.reshape(subcategory, (-1,)))
        subcategory_vector = self.subcategory_dense(subcategory_embedded)
        # title
        title_embedded = self.word_embedding(ops.reshape(title, self.title_shape))
        title_feature = self.title_CNN(ops.Transpose()(title_embedded, (0, 2, 1)))
        title_vector = self.title_attention(ops.Transpose()(title_feature, (0, 2, 1)))
        # abstract
        abstract_embedded = self.word_embedding(ops.reshape(abstract, self.abstract_shape))
        abstract_feature = self.abstract_CNN(ops.Transpose()(abstract_embedded, (0, 2, 1)))
        abstract_vector = self.abstract_attention(ops.Transpose()(abstract_feature, (0, 2, 1)))
        # total
        news_vector = self.total_attention(
            self.pack((category_vector, subcategory_vector, title_vector, abstract_vector)))
        return news_vector

class UserEncoder(nn.Cell):
    """
    The main function to create user encoder of NAML.

    Args:
        args (class): global hyper-parameters.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> UserEncoder(args)
    """
    def __init__(self, args):
        super(UserEncoder, self).__init__()
        self.news_attention = Attention(args.query_vector_dim, args.n_filters)

    def construct(self, news_vectors):
        user_vector = self.news_attention(news_vectors)
        return user_vector

class ClickPredictor(nn.Cell):
    """
    Click predictor by user encoder and news encoder.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ClickPredictor()
    """
    def __init__(self):
        super(ClickPredictor, self).__init__()
        self.matmul = ops.BatchMatMul()

    def construct(self, news_vector, user_vector):
        predict = ops.Flatten()(self.matmul(news_vector, ops.expand_dims(user_vector, 2)))
        return predict

class NAML(nn.Cell):
    """
    NAML model(Neural News Recommendation with Attentive Multi-View Learning).

    Args:
        args (class): global hyper-parameters.
        word_embedding (Tensor): parameter of word embedding.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> NAML(rgs, embedding_table)
    """
    def __init__(self, args, embedding_table=None):
        super(NAML, self).__init__()
        self.args = args
        self.news_encoder = NewsEncoder(args, embedding_table)
        self.user_encoder = UserEncoder(args)
        self.click_predictor = ClickPredictor()
        self.browsed_vector_shape = (args.batch_size, args.n_browsed_news, args.n_filters)
        self.candidate_vector_shape = (args.batch_size, args.neg_sample + 1, args.n_filters)
        if not embedding_table is None:
            self.word_embedding_shape = embedding_table.shape
        else:
            self.word_embedding_shape = ()
        self._initialize_weights()

    def construct(self, category_b, subcategory_b, title_b, abstract_b, category_c, subcategory_c, title_c, abstract_c):
        browsed_news_vectors = ops.reshape(self.news_encoder(category_b, subcategory_b, title_b, abstract_b),
                                           self.browsed_vector_shape)
        user_vector = self.user_encoder(browsed_news_vectors)
        candidate_news_vector = ops.reshape(self.news_encoder(category_c, subcategory_c, title_c, abstract_c),
                                            self.candidate_vector_shape)
        predict = self.click_predictor(candidate_news_vector, user_vector)
        return predict

    def _initialize_weights(self):
        """Weights initialize."""
        self.init_parameters_data()
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv1d):
                cell.weight.set_data(init.initializer("XavierUniform",
                                                      cell.weight.shape,
                                                      cell.weight.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(init.initializer("XavierUniform",
                                                      cell.weight.shape,
                                                      cell.weight.dtype))
            elif isinstance(cell, nn.Embedding) and cell.embedding_table.shape != self.word_embedding_shape:
                cell.embedding_table.set_data(init.initializer("uniform",
                                                               cell.embedding_table.shape,
                                                               cell.embedding_table.dtype))

class NAMLWithLossCell(nn.Cell):
    """
    NAML add loss Cell.

    Args:
        network (Cell): naml network.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> NAMLWithLossCell(NAML(rgs, word_embedding))
    """
    def __init__(self, network):
        super(NAMLWithLossCell, self).__init__()
        self.network = network
        self.loss = nn.SoftmaxCrossEntropyWithLogits(sparse=False, reduction='mean')

    def construct(self, category_b, subcategory_b, title_b, abstract_b, category_c, subcategory_c, title_c, abstract_c,
                  label):
        predict = self.network(category_b, subcategory_b, title_b, abstract_b, category_c, subcategory_c, title_c,
                               abstract_c)
        dtype = ops.dtype(predict)
        shp = ops.shape(predict)
        loss = self.loss(predict, ops.reshape(ops.cast(label, dtype), shp))
        return loss

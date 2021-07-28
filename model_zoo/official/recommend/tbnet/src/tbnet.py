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
"""TB-Net Model."""

from mindspore import nn
from mindspore import ParameterTuple
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.parallel._utils import _get_device_num, _get_parallel_mode, _get_gradients_mean
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer

from src.embedding import EmbeddingMatrix


class TBNet(nn.Cell):
    """
    TB-Net model architecture.

    Args:
        num_entity (int): number of entities, depends on dataset
        num_relation (int): number of relations, depends on dataset
        dim (int): dimension of entity and relation embedding vectors
        kge_weight (float): weight of the KG Embedding loss term
        node_weight (float): weight of the node loss term (default=0.002)
        l2_weight (float): weight of the L2 regularization term (default=1e-7)
        lr (float): learning rate of model training (default=1e-4)
        batch_size (int): batch size (default=1024)
    """

    def __init__(self, config):
        super(TBNet, self).__init__()

        self._parse_config(config)
        self.matmul = C.matmul
        self.sigmoid = P.Sigmoid()
        self.embedding_initializer = "normal"

        self.entity_emb_matrix = EmbeddingMatrix(int(self.num_entity),
                                                 self.dim,
                                                 embedding_table=self.embedding_initializer)
        self.relation_emb_matrix = EmbeddingMatrix(int(self.num_relation),
                                                   embedding_size=(self.dim, self.dim),
                                                   embedding_table=self.embedding_initializer)

        self.expand_dims = P.ExpandDims()
        self.squeeze = P.Squeeze(3)
        self.abs = P.Abs()
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()

        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        self.softmax = nn.Softmax()

    def _parse_config(self, config):
        """Argument parsing."""

        self.num_entity = config.num_entity
        self.num_relation = config.num_relation
        self.dim = config.embedding_dim
        self.kge_weight = config.kge_weight
        self.node_weight = config.node_weight
        self.l2_weight = config.l2_weight
        self.lr = config.lr
        self.batch_size = config.batch_size

    def construct(self, items, relation1, mid_entity, relation2, hist_item):
        """
        TB-Net main computation process.

        Args:
            items (Tensor): rated item IDs, int Tensor in shape of [batch size, ].
            relation1 (Tensor): relation1 IDs, int Tensor in shape of [batch size, per_item_num_paths]
            mid_entity (Tensor): middle entity IDs, int Tensor in shape of [batch size, per_item_num_paths]
            relation2 (Tensor): relation2 IDs, int Tensor in shape of [batch size, per_item_num_paths]
            hist_item (Tensor): historical item IDs, int Tensor in shape of [batch size, per_item_num_paths]

        Returns:
            scores (Tensor): model prediction score, float Tensor in shape of [batch size, ]
            probs_exp (Tensor): path probability/importance, float Tensor in shape of [batch size, per_item_num_paths]
            item_embeddings (Tensor): rated item embeddings, float Tensor in shape of [batch size, dim]
            relation1_emb (Tensor): relation1 embeddings,
                                    float Tensor in shape of [batch size, per_item_num_paths, dim, dim]
            mid_entity_emb (Tensor): middle entity embeddings,
                                     float Tensor in shape of [batch size, per_item_num_paths, dim]
            relation2_emb (Tensor): relation2 embeddings,
                                    float Tensor in shape of [batch size, per_item_num_paths, dim, dim]
            hist_item_emb (Tensor): historical item embeddings,
                                    float Tensor in shape of [batch size, per_item_num_paths, dim]
        """
        item_embeddings = self.entity_emb_matrix(items)

        relation1_emb = self.relation_emb_matrix(relation1)
        mid_entity_emb = self.entity_emb_matrix(mid_entity)
        relation2_emb = self.relation_emb_matrix(relation2)
        hist_item_emb = self.entity_emb_matrix(hist_item)

        response, probs_exp = self._key_pathing(item_embeddings,
                                                relation1_emb,
                                                mid_entity_emb,
                                                relation2_emb,
                                                hist_item_emb)

        scores = P.Squeeze()(self._predict(item_embeddings, response))

        return scores, probs_exp, item_embeddings, relation1_emb, mid_entity_emb, relation2_emb, hist_item_emb

    def _key_pathing(self, item_embeddings, relation1_emb, mid_entity_emb, relation2_emb, hist_item_emb):
        """
        Compute the response and path probability using item and entity embedding.
        Path structure: (rated item, relation1, entity, relation2, historical item).

        Args:
            item_embeddings (Tensor): rated item embeddings, float Tensor in shape of [batch size, dim]
            relation1_emb (Tensor): relation1 embeddings,
                                    float Tensor in shape of [batch size, per_item_num_paths, dim, dim]
            mid_entity_emb (Tensor): middle entity embeddings,
                                     float Tensor in shape of [batch size, per_item_num_paths, dim]
            relation2_emb (Tensor): relation2 embeddings,
                                    float Tensor in shape of [batch size, per_item_num_paths, dim, dim]
            hist_item_emb (Tensor): historical item embeddings,
                                    float Tensor in shape of [batch size, per_item_num_paths, dim]

        Returns:
            response (Tensor): user's response towards middle entity, float Tensor in shape of [batch size, dim]
            probs_exp (Tensor): path probability/importance, float Tensor in shape of [batch size, per_item_num_paths]
        """

        hist_item_e_4d = self.expand_dims(hist_item_emb, 3)
        mul_r2_hist = self.squeeze(self.matmul(relation2_emb, hist_item_e_4d))
        # path_right shape: [batch size, per_item_num_paths, dim]
        path_right = self.abs(mul_r2_hist + self.reduce_sum(relation2_emb, 2))

        item_emb_3d = self.expand_dims(item_embeddings, 2)
        mul_r1_item = self.squeeze(self.matmul(relation1_emb, self.expand_dims(item_emb_3d, 1)))
        path_left = self.abs(mul_r1_item + self.reduce_sum(relation1_emb, 2))
        # path_left shape: [batch size, dim, per_item_num_paths]
        path_left = self.transpose(path_left, (0, 2, 1))

        probs = self.reduce_sum(self.matmul(path_right, path_left), 2)
        # probs_exp shape: [batch size, per_item_num_paths]
        probs_exp = self.softmax(probs)

        probs_3d = self.expand_dims(probs_exp, 2)
        # response shape: [batch size, dim]
        response = self.reduce_sum(mid_entity_emb * probs_3d, 1)

        return response, probs_exp

    def _predict(self, item_embeddings, response):
        scores = self.reduce_sum(item_embeddings * response, 1)

        return scores


class NetWithLossClass(nn.Cell):
    """NetWithLossClass definition."""

    def __init__(self, network, config):
        super(NetWithLossClass, self).__init__(auto_prefix=False)
        self.network = network
        self.loss = P.SigmoidCrossEntropyWithLogits()
        self.matmul = C.matmul
        self.expand_dims = P.ExpandDims()
        self.squeeze = P.Squeeze(3)
        self.abs = P.Abs()
        self.maximum = P.Maximum()
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.sigmoid = P.Sigmoid()

        self.kge_weight = config.kge_weight
        self.node_weight = config.node_weight
        self.l2_weight = config.l2_weight
        self.batch_size = config.batch_size
        self.dim = config.embedding_dim

        self.embedding_initializer = "normal"

    def construct(self, items, relation1, mid_entity, relation2, hist_item, labels):
        """
        Args:
            items (Tensor): rated item IDs, int Tensor in shape of [batch size, ].
            relation1 (Tensor): relation1 IDs, int Tensor in shape of [batch size, per_item_num_paths]
            mid_entity (Tensor): middle entity IDs, int Tensor in shape of [batch size, per_item_num_paths]
            relation2 (Tensor): relation2 IDs, int Tensor in shape of [batch size, per_item_num_paths]
            hist_item (Tensor): historical item IDs, int Tensor in shape of [batch size, per_item_num_paths]
            labels (Tensor): label of rated item record, int Tensor in shape of [batch size, ]

        Returns:
            loss (float): loss value
        """
        scores, _, item_embeddings, relation1_emb, mid_entity_emb, relation2_emb, hist_item_emb = \
            self.network(items, relation1, mid_entity, relation2, hist_item)
        loss = self._loss_fun(item_embeddings, relation1_emb, mid_entity_emb,
                              relation2_emb, hist_item_emb, scores, labels)

        return loss

    def _loss_fun(self, item_embeddings, relation1_emb, mid_entity_emb, relation2_emb, hist_item_emb, scores, labels):
        """
        Loss function definition.

        Args:
            item_embeddings (Tensor): rated item embeddings, float Tensor in shape of [batch size, dim]
            relation1_emb (Tensor): relation1 embeddings,
                                    float Tensor in shape of [batch size, per_item_num_paths, dim, dim]
            mid_entity_emb (Tensor): middle entity embeddings,
                                     float Tensor in shape of [batch size, per_item_num_paths, dim]
            relation2_emb (Tensor): relation2 embeddings,
                                    float Tensor in shape of [batch size, per_item_num_paths, dim, dim]
            hist_item_emb (Tensor): historical item embeddings,
                                    float Tensor in shape of [batch size, per_item_num_paths, dim]
            scores (Tensor): model prediction score, float Tensor in shape of [batch size, ]
            labels (Tensor): label of rated item record, int Tensor in shape of [batch size, ]
        Returns:
            loss: includes four part:
                pred_loss: cross entropy of the model prediction score and labels
                transr_loss: TransR KG Embedding loss
                node_loss: node matching loss
                l2_loss: L2 regularization loss
        """
        pred_loss = self.reduce_mean(self.loss(scores, labels))

        item_emb_3d = self.expand_dims(item_embeddings, 2)
        item_emb_4d = self.expand_dims(item_emb_3d, 1)

        mul_r1_item = self.squeeze(self.matmul(relation1_emb, item_emb_4d))

        hist_item_e_4d = self.expand_dims(hist_item_emb, 3)
        mul_r2_hist = self.squeeze(self.matmul(relation2_emb, hist_item_e_4d))

        relation1_3d = self.reduce_sum(relation1_emb, 2)
        relation2_3d = self.reduce_sum(relation2_emb, 2)

        path_left = self.reduce_sum(self.abs(mul_r1_item + relation1_3d), 2)
        path_right = self.reduce_sum(self.abs(mul_r2_hist + relation2_3d), 2)

        transr_loss = self.reduce_sum(self.maximum(self.abs(path_left - path_right), 0))
        transr_loss = self.reduce_mean(self.sigmoid(transr_loss))

        mid_entity_emb_4d = self.expand_dims(mid_entity_emb, 3)
        mul_r2_mid = self.squeeze(self.matmul(relation2_emb, mid_entity_emb_4d))
        path_r2_mid = self.abs(mul_r2_mid + relation2_3d)

        node_loss = self.reduce_sum(self.maximum(mul_r2_hist - path_r2_mid, 0))
        node_loss = self.reduce_mean(self.sigmoid(node_loss))

        l2_loss = self.reduce_mean(self.reduce_sum(relation1_emb * relation1_emb))
        l2_loss += self.reduce_mean(self.reduce_sum(mid_entity_emb * mid_entity_emb))
        l2_loss += self.reduce_mean(self.reduce_sum(relation2_emb * relation2_emb))
        l2_loss += self.reduce_mean(self.reduce_sum(hist_item_emb * hist_item_emb))

        transr_loss = self.kge_weight * transr_loss
        node_loss = self.node_weight * node_loss

        l2_loss = self.l2_weight * l2_loss

        loss = pred_loss + transr_loss + node_loss + l2_loss

        return loss


class TrainStepWrap(nn.Cell):
    """TrainStepWrap definition."""

    def __init__(self, network, lr, sens=1):
        super(TrainStepWrap, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_train()
        self.network.add_flags(defer_inline=True)
        self.weights = ParameterTuple(network.trainable_params())
        self.lr = lr
        self.optimizer = nn.Adam(self.weights,
                                 learning_rate=self.lr,
                                 beta1=0.9,
                                 beta2=0.999,
                                 eps=1e-8,
                                 loss_scale=sens)

        self.hyper_map = C.HyperMap()
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens

        self.reducer_flag = False
        self.grad_reducer = None
        parallel_mode = _get_parallel_mode()
        if parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(self.optimizer.parameters, mean, degree)

    def construct(self, items, relation1, mid_entity, relation2, hist_item, labels):
        """
        Args:
            items (Tensor): rated item IDs, int Tensor in shape of [batch size, ].
            relation1 (Tensor): relation1 IDs, int Tensor in shape of [batch size, per_item_num_paths]
            mid_entity (Tensor): middle entity IDs, int Tensor in shape of [batch size, per_item_num_paths]
            relation2 (Tensor): relation2 IDs, int Tensor in shape of [batch size, per_item_num_paths]
            hist_item (Tensor): historical item IDs, int Tensor in shape of [batch size, per_item_num_paths]
            labels (Tensor): label of rated item record, int Tensor in shape of [batch size, ]

        Returns:
            loss and gradient
        """
        weights = self.weights
        loss = self.network(items, relation1, mid_entity, relation2, hist_item, labels)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(items, relation1, mid_entity, relation2, hist_item, labels, sens)

        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)

        return F.depend(loss, self.optimizer(grads))


class PredictWithSigmoid(nn.Cell):
    """Predict method."""

    def __init__(self, network):
        super(PredictWithSigmoid, self).__init__(auto_prefix=False)
        self.network = network
        self.sigmoid = P.Sigmoid()

    def construct(self, items, relation1, mid_entity, relation2, hist_item, labels):
        """
        Predict with sigmoid definition.

        Args:
            items (Tensor): rated item IDs, int Tensor in shape of [batch size, ].
            relation1 (Tensor): relation1 IDs, int Tensor in shape of [batch size, per_item_num_paths]
            mid_entity (Tensor): middle entity IDs, int Tensor in shape of [batch size, per_item_num_paths]
            relation2 (Tensor): relation2 IDs, int Tensor in shape of [batch size, per_item_num_paths]
            hist_item (Tensor): historical item IDs, int Tensor in shape of [batch size, per_item_num_paths]
            labels (Tensor): label of rated item record, int Tensor in shape of [batch size, ]

        Returns:
            scores (Tensor): model prediction score, float Tensor in shape of [batch size, ]
            pred_probs (Tensor): prediction probability, float Tensor in shape of [batch size, ]
            labels (Tensor): label of rated item record, int Tensor in shape of [batch size, ]
            probs_exp (Tensor): path probability/importance, float Tensor in shape of [batch size, per_item_num_paths]
        """

        scores, probs_exp, _, _, _, _, _ = self.network(items, relation1, mid_entity, relation2, hist_item)
        pred_probs = self.sigmoid(scores)

        return scores, pred_probs, labels, probs_exp

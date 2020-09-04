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
"""
callback
"""
import numpy as np

from mindspore import nn
from mindspore import Tensor
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from mindspore.common.parameter import ParameterTuple

from src.utils import convert_item_id


def TestBGCF(forward_net, num_user, num_item, input_dim, test_graph_dataset):
    """BGCF test wrapper"""
    user_reps = np.zeros([num_user, input_dim * 3])
    item_reps = np.zeros([num_item, input_dim * 3])

    for _ in range(50):
        test_graph_dataset.random_select_sampled_graph()
        u_test_neighs, u_test_gnew_neighs = test_graph_dataset.get_user_sapmled_neighbor()
        i_test_neighs, i_test_gnew_neighs = test_graph_dataset.get_item_sampled_neighbor()

        u_test_neighs = Tensor(convert_item_id(u_test_neighs, num_user), mstype.int32)
        u_test_gnew_neighs = Tensor(convert_item_id(u_test_gnew_neighs, num_user), mstype.int32)
        i_test_neighs = Tensor(i_test_neighs, mstype.int32)
        i_test_gnew_neighs = Tensor(i_test_gnew_neighs, mstype.int32)

        users = Tensor(np.arange(num_user).reshape(-1,), mstype.int32)
        items = Tensor(np.arange(num_item).reshape(-1,), mstype.int32)
        neg_items = Tensor(np.arange(num_item).reshape(-1, 1), mstype.int32)

        user_rep, item_rep = forward_net(users,
                                         items,
                                         neg_items,
                                         u_test_neighs,
                                         u_test_gnew_neighs,
                                         i_test_neighs,
                                         i_test_gnew_neighs)

        user_reps += user_rep.asnumpy()
        item_reps += item_rep.asnumpy()

    user_reps /= 50
    item_reps /= 50
    return user_reps, item_reps


class ForwardBGCF(nn.Cell):
    """Calculate the forward output"""

    def __init__(self,
                 network):
        super(ForwardBGCF, self).__init__()
        self.network = network

    def construct(self, users, items, neg_items, u_neighs, u_gnew_neighs, i_neighs, i_gnew_neighs):
        """Calculate the user and item representation"""
        _, user_rep, _, item_rep, _, _, = self.network(users,
                                                       items,
                                                       neg_items,
                                                       users,
                                                       items,
                                                       users,
                                                       u_neighs,
                                                       u_gnew_neighs,
                                                       items,
                                                       i_neighs,
                                                       i_gnew_neighs,
                                                       items,
                                                       i_neighs,
                                                       i_gnew_neighs,
                                                       1)
        return user_rep, item_rep


class BGCFLoss(nn.Cell):
    """BGCF loss with user and item embedding"""

    def __init__(self, neg_item_num, l2_embed, dist_reg):
        super(BGCFLoss, self).__init__()

        self.neg_item_num = neg_item_num
        self.l2_embed = l2_embed
        self.dist_reg = dist_reg

        self.log = P.Log()
        self.pow = P.Pow()
        self.cast = P.Cast()
        self.tile = P.Tile()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.concat = P.Concat(1)
        self.concat2 = P.Concat(2)
        self.split = P.Split(0, 2)
        self.reduce_sum = P.ReduceSum()
        self.expand_dims = P.ExpandDims()
        self.multiply = P.Mul()
        self.matmul = P.BatchMatMul()
        self.squeeze = P.Squeeze(1)
        self.transpose = P.Transpose()
        self.l2_loss = P.L2Loss()
        self.sigmoid = P.Sigmoid()

    def construct(self, all_user_embed, all_user_rep, all_pos_item_embed,
                  all_pos_item_rep, neg_item_embed, neg_item_rep):
        """Calculate loss"""
        all_user_embed = self.cast(all_user_embed, mstype.float16)
        all_user_rep = self.concat((all_user_rep, all_user_embed))

        user_rep, pos_user_rep = self.split(all_user_rep)
        user_embed, pos_user_embed = self.split(all_user_embed)

        user_user_distance = self.reduce_sum(self.pow(user_rep - pos_user_rep, 2)) \
                             + self.reduce_sum(self.pow(user_embed - pos_user_embed, 2))
        user_user_distance = self.cast(user_user_distance, mstype.float32)

        user_rep = self.expand_dims(user_rep, 1)

        all_pos_item_embed = self.cast(all_pos_item_embed, mstype.float16)
        all_pos_item_rep = self.concat((all_pos_item_rep, all_pos_item_embed))

        pos_item_rep, pos_item_neigh_rep = self.split(all_pos_item_rep)
        pos_item_embed, pos_item_neigh_embed = self.split(all_pos_item_embed)

        pos_item_item_distance = self.reduce_sum(self.pow(pos_item_rep - pos_item_neigh_rep, 2)) \
                                 + self.reduce_sum(self.pow(pos_item_embed - pos_item_neigh_embed, 2))
        pos_item_item_distance = self.cast(pos_item_item_distance, mstype.float32)

        neg_item_embed = self.cast(neg_item_embed, mstype.float16)
        neg_item_rep = self.concat2((neg_item_rep, neg_item_embed))

        item_rep = self.concat((self.expand_dims(pos_item_rep, 1), neg_item_rep))

        pos_rating = self.reduce_sum(self.multiply(self.squeeze(user_rep), pos_item_rep), 1)
        pos_rating = self.expand_dims(pos_rating, 1)
        pos_rating = self.tile(pos_rating, (1, self.neg_item_num))
        pos_rating = self.reshape(pos_rating, (self.shape(pos_rating)[0] * self.neg_item_num, 1))
        pos_rating = self.cast(pos_rating, mstype.float32)

        batch_neg_item_embedding = self.transpose(neg_item_rep, (0, 2, 1))
        neg_rating = self.matmul(user_rep, batch_neg_item_embedding)
        neg_rating = self.squeeze(neg_rating)
        neg_rating = self.reshape(neg_rating, (self.shape(neg_rating)[0] * self.neg_item_num, 1))
        neg_rating = self.cast(neg_rating, mstype.float32)

        bpr_loss = pos_rating - neg_rating
        bpr_loss = self.sigmoid(bpr_loss)
        bpr_loss = - self.log(bpr_loss)
        bpr_loss = self.reduce_sum(bpr_loss)

        reg_loss = self.l2_embed * (self.l2_loss(user_rep) + self.l2_loss(item_rep))

        loss = bpr_loss + reg_loss + self.dist_reg * (user_user_distance + pos_item_item_distance)
        return loss


class LossWrapper(nn.Cell):
    """
    Wraps the BGCF model with loss.

    Args:
        network (Cell): BGCF network.
        neg_item_num (Number): The num of negative instances for a positive instance.
        l2_embed (Number): The coefficient of l2 loss.
        dist_reg (Number): The coefficient of distance loss.
    """

    def __init__(self, network, neg_item_num, l2_embed, dist_reg=0.002):
        super(LossWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.loss_func = BGCFLoss(neg_item_num, l2_embed, dist_reg)

    def construct(self,
                  u_id,
                  pos_item_id,
                  neg_item_id,
                  pos_users,
                  pos_items,
                  u_group_nodes,
                  u_neighs,
                  u_gnew_neighs,
                  i_group_nodes,
                  i_neighs,
                  i_gnew_neighs,
                  neg_group_nodes,
                  neg_neighs,
                  neg_gnew_neighs):
        """Return loss"""
        all_user_embed, all_user_rep, all_pos_item_embed, \
        all_pos_item_rep, neg_item_embed, neg_item_rep = self.network(u_id,
                                                                      pos_item_id,
                                                                      neg_item_id,
                                                                      pos_users,
                                                                      pos_items,
                                                                      u_group_nodes,
                                                                      u_neighs,
                                                                      u_gnew_neighs,
                                                                      i_group_nodes,
                                                                      i_neighs,
                                                                      i_gnew_neighs,
                                                                      neg_group_nodes,
                                                                      neg_neighs,
                                                                      neg_gnew_neighs,
                                                                      10)
        loss = self.loss_func(all_user_embed, all_user_rep, all_pos_item_embed,
                              all_pos_item_rep, neg_item_embed, neg_item_rep)
        return loss


class TrainOneStepCell(nn.Cell):
    r"""
    Network training package class.

    Wraps the network with an optimizer. The resulting Cell be trained with sample inputs.
    Backward graph will be created in the construct function to do parameter updating. Different
    parallel models are available to run the training.

    Args:
        network (Cell): The training network.
        optimizer (Cell): Optimizer for updating the weights.
        sens (Number): The scaling number to be filled as the input of backpropagation. Default value is 1.0.

    Outputs:
        Tensor, a scalar Tensor with shape :math:`()`.

    Examples:
        >>> net = Net()
        >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits()
        >>> optim = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> loss_net = nn.WithLossCell(net, loss_fn)
        >>> train_net = nn.TrainOneStepCell(loss_net, optim)
    """

    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)

        self.network = network
        self.network.add_flags(defer_inline=True)
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens

    def construct(self,
                  u_id,
                  pos_item_id,
                  neg_item_id,
                  pos_users,
                  pos_items,
                  u_group_nodes,
                  u_neighs,
                  u_gnew_neighs,
                  i_group_nodes,
                  i_neighs,
                  i_gnew_neighs,
                  neg_group_nodes,
                  neg_neighs,
                  neg_gnew_neighs):
        """Grad process"""
        weights = self.weights
        loss = self.network(u_id,
                            pos_item_id,
                            neg_item_id,
                            pos_users,
                            pos_items,
                            u_group_nodes,
                            u_neighs,
                            u_gnew_neighs,
                            i_group_nodes,
                            i_neighs,
                            i_gnew_neighs,
                            neg_group_nodes,
                            neg_neighs,
                            neg_gnew_neighs)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(u_id,
                                                 pos_item_id,
                                                 neg_item_id,
                                                 pos_users,
                                                 pos_items,
                                                 u_group_nodes,
                                                 u_neighs,
                                                 u_gnew_neighs,
                                                 i_group_nodes,
                                                 i_neighs,
                                                 i_gnew_neighs,
                                                 neg_group_nodes,
                                                 neg_neighs,
                                                 neg_gnew_neighs,
                                                 sens)
        return F.depend(loss, self.optimizer(grads))


class TrainBGCF(nn.Cell):
    """
       Wraps the BGCF model with optimizer.

       Args:
           network (Cell): BGCF network.
           neg_item_num (Number): The num of negative instances for a positive instance.
           l2_embed (Number): The coefficient of l2 loss.
           learning_rate (Number): The learning rate.
           epsilon (Number):The term added to the denominator to improve numerical stability.
           dist_reg (Number): The coefficient of distance loss.
    """

    def __init__(self,
                 network,
                 neg_item_num,
                 l2_embed,
                 learning_rate,
                 epsilon,
                 dist_reg=0.002):
        super(TrainBGCF, self).__init__(auto_prefix=False)

        self.network = network
        loss_net = LossWrapper(network,
                               neg_item_num,
                               l2_embed,
                               dist_reg)
        optimizer = nn.Adam(loss_net.trainable_params(),
                            learning_rate=learning_rate,
                            eps=epsilon)
        self.loss_train_net = TrainOneStepCell(loss_net, optimizer)

    def construct(self,
                  u_id,
                  pos_item_id,
                  neg_item_id,
                  pos_users,
                  pos_items,
                  u_group_nodes,
                  u_neighs,
                  u_gnew_neighs,
                  i_group_nodes,
                  i_neighs,
                  i_gnew_neighs,
                  neg_group_nodes,
                  neg_neighs,
                  neg_gnew_neighs):
        """Return loss"""
        loss = self.loss_train_net(u_id,
                                   pos_item_id,
                                   neg_item_id,
                                   pos_users,
                                   pos_items,
                                   u_group_nodes,
                                   u_neighs,
                                   u_gnew_neighs,
                                   i_group_nodes,
                                   i_neighs,
                                   i_gnew_neighs,
                                   neg_group_nodes,
                                   neg_neighs,
                                   neg_gnew_neighs)
        return loss

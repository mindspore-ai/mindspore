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
"""Architecture"""
import os
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import Parameter, Tensor, context
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.train.serialization import export

context.set_context(mode=context.PYNATIVE_MODE)


class MeanConv(nn.Cell):
    def __init__(self,
                 feature_in_dim,
                 feature_out_dim,
                 activation,
                 dropout=0.2):
        super(MeanConv, self).__init__()
        self.out_weight = Parameter(
            initializer("XavierUniform", [feature_in_dim * 2, feature_out_dim], dtype=mstype.float32))
        if activation == "tanh":
            self.act = P.Tanh()
        elif activation == "relu":
            self.act = P.ReLU()
        else:
            raise ValueError("activation should be tanh or relu")
        self.cast = P.Cast()
        self.matmul = P.MatMul()
        self.concat = P.Concat(axis=1)
        self.reduce_mean = P.ReduceMean(keep_dims=False)
        self.dropout = nn.Dropout(p=dropout)

    def construct(self, self_feature, neigh_feature):
        neigh_matrix = self.reduce_mean(neigh_feature, 1)
        neigh_matrix = self.dropout(neigh_matrix)
        output = self.concat((self_feature, neigh_matrix))
        output = self.act(self.matmul(output, self.out_weight))
        return output


class AttenConv(nn.Cell):
    def __init__(self,
                 feature_in_dim,
                 feature_out_dim,
                 dropout=0.2):
        super(AttenConv, self).__init__()
        self.out_weight = Parameter(
            initializer("XavierUniform", [feature_in_dim * 2, feature_out_dim], dtype=mstype.float32))
        self.cast = P.Cast()
        self.squeeze = P.Squeeze(1)
        self.concat = P.Concat(axis=1)
        self.expanddims = P.ExpandDims()
        self.softmax = P.Softmax(axis=-1)
        self.matmul = P.MatMul()
        self.matmul_3 = P.BatchMatMul()
        self.matmul_t = P.BatchMatMul(transpose_b=True)
        self.dropout = nn.Dropout(p=dropout)

    def construct(self, self_feature, neigh_feature):
        query = self.expanddims(self_feature, 1)
        neigh_matrix = self.dropout(neigh_feature)
        score = self.matmul_t(query, neigh_matrix)
        score = self.softmax(score)
        atten_agg = self.matmul_3(score, neigh_matrix)
        atten_agg = self.squeeze(atten_agg)
        output = self.matmul(self.concat((atten_agg, self_feature)), self.out_weight)
        return output


class BGCF(nn.Cell):
    def __init__(self,
                 dataset_argv,
                 architect_argv,
                 activation,
                 neigh_drop_rate,
                 num_user,
                 num_item,
                 input_dim):
        super(BGCF, self).__init__()
        self.user_embed = Parameter(initializer("XavierUniform", [num_user, input_dim], dtype=mstype.float32))
        self.item_embed = Parameter(initializer("XavierUniform", [num_item, input_dim], dtype=mstype.float32))
        self.cast = P.Cast()
        self.tanh = P.Tanh()
        self.shape = P.Shape()
        self.split = P.Split(0, 2)
        self.gather = P.Gather()
        self.reshape = P.Reshape()
        self.concat_0 = P.Concat(0)
        self.concat_1 = P.Concat(1)
        (self.input_dim, self.num_user, self.num_item) = dataset_argv
        self.layer_dim = architect_argv
        self.gnew_agg_mean = MeanConv(self.input_dim, self.layer_dim,
                                      activation=activation, dropout=neigh_drop_rate[1])
        self.gnew_agg_mean.to_float(mstype.float16)
        self.gnew_agg_user = AttenConv(self.input_dim, self.layer_dim, dropout=neigh_drop_rate[2])
        self.gnew_agg_user.to_float(mstype.float16)
        self.gnew_agg_item = AttenConv(self.input_dim, self.layer_dim, dropout=neigh_drop_rate[2])
        self.gnew_agg_item.to_float(mstype.float16)
        self.user_feature_dim = self.input_dim
        self.item_feature_dim = self.input_dim
        self.final_weight = Parameter(
            initializer("XavierUniform", [self.input_dim * 3, self.input_dim * 3], dtype=mstype.float32))
        self.raw_agg_funcs_user = MeanConv(self.input_dim, self.layer_dim,
                                           activation=activation, dropout=neigh_drop_rate[0])
        self.raw_agg_funcs_user.to_float(mstype.float16)
        self.raw_agg_funcs_item = MeanConv(self.input_dim, self.layer_dim,
                                           activation=activation, dropout=neigh_drop_rate[0])
        self.raw_agg_funcs_item.to_float(mstype.float16)

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
                  neg_gnew_neighs,
                  neg_item_num):
        all_user_embed = self.gather(self.user_embed, self.concat_0((u_id, pos_users)), 0)
        u_self_matrix_at_layers = self.gather(self.user_embed, u_group_nodes, 0)
        u_neigh_matrix_at_layers = self.gather(self.item_embed, u_neighs, 0)
        u_output_mean = self.raw_agg_funcs_user(u_self_matrix_at_layers, u_neigh_matrix_at_layers)
        u_gnew_neighs_matrix = self.gather(self.item_embed, u_gnew_neighs, 0)
        u_output_from_gnew_mean = self.gnew_agg_mean(u_self_matrix_at_layers, u_gnew_neighs_matrix)
        u_output_from_gnew_att = self.gnew_agg_user(u_self_matrix_at_layers,
                                                    self.concat_1((u_neigh_matrix_at_layers, u_gnew_neighs_matrix)))
        u_output = self.concat_1((u_output_mean, u_output_from_gnew_mean, u_output_from_gnew_att))
        all_user_rep = self.tanh(u_output)
        all_pos_item_embed = self.gather(self.item_embed, self.concat_0((pos_item_id, pos_items)), 0)
        i_self_matrix_at_layers = self.gather(self.item_embed, i_group_nodes, 0)
        i_neigh_matrix_at_layers = self.gather(self.user_embed, i_neighs, 0)
        i_output_mean = self.raw_agg_funcs_item(i_self_matrix_at_layers, i_neigh_matrix_at_layers)
        i_gnew_neighs_matrix = self.gather(self.user_embed, i_gnew_neighs, 0)
        i_output_from_gnew_mean = self.gnew_agg_mean(i_self_matrix_at_layers, i_gnew_neighs_matrix)
        i_output_from_gnew_att = self.gnew_agg_item(i_self_matrix_at_layers,
                                                    self.concat_1((i_neigh_matrix_at_layers, i_gnew_neighs_matrix)))
        i_output = self.concat_1((i_output_mean, i_output_from_gnew_mean, i_output_from_gnew_att))
        all_pos_item_rep = self.tanh(i_output)
        neg_item_embed = self.gather(self.item_embed, neg_item_id, 0)
        neg_self_matrix_at_layers = self.gather(self.item_embed, neg_group_nodes, 0)
        neg_neigh_matrix_at_layers = self.gather(self.user_embed, neg_neighs, 0)
        neg_output_mean = self.raw_agg_funcs_item(neg_self_matrix_at_layers, neg_neigh_matrix_at_layers)
        neg_gnew_neighs_matrix = self.gather(self.user_embed, neg_gnew_neighs, 0)
        neg_output_from_gnew_mean = self.gnew_agg_mean(neg_self_matrix_at_layers, neg_gnew_neighs_matrix)
        neg_output_from_gnew_att = self.gnew_agg_item(neg_self_matrix_at_layers,
                                                      self.concat_1(
                                                          (neg_neigh_matrix_at_layers, neg_gnew_neighs_matrix)))
        neg_output = self.concat_1((neg_output_mean, neg_output_from_gnew_mean, neg_output_from_gnew_att))
        neg_output = self.tanh(neg_output)
        neg_output_shape = self.shape(neg_output)
        neg_item_rep = self.reshape(neg_output,
                                    (self.shape(neg_item_embed)[0], neg_item_num, neg_output_shape[-1]))

        return all_user_embed, all_user_rep, all_pos_item_embed, all_pos_item_rep, neg_item_embed, neg_item_rep


class ForwardBGCF(nn.Cell):
    def __init__(self,
                 network):
        super(ForwardBGCF, self).__init__()
        self.network = network

    def construct(self, users, items, neg_items, u_neighs, u_gnew_neighs, i_neighs, i_gnew_neighs):
        _, user_rep, _, item_rep, _, _, = self.network(users, items, neg_items, users, items, users,
                                                       u_neighs, u_gnew_neighs, items, i_neighs, i_gnew_neighs,
                                                       items, i_neighs, i_gnew_neighs, 1)
        return user_rep, item_rep

@pytest.mark.level1
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_export_bgcf():
    context.set_context(mode=context.GRAPH_MODE)
    num_user, num_item = 7068, 3570
    network = BGCF([64, num_user, num_item], 64, "tanh",
                   [0.0, 0.0, 0.0], num_user, num_item, 64)

    forward_net = ForwardBGCF(network)
    users = Tensor(np.zeros([num_user,]).astype(np.int32))
    items = Tensor(np.zeros([num_item,]).astype(np.int32))
    neg_items = Tensor(np.zeros([num_item, 1]).astype(np.int32))
    u_test_neighs = Tensor(np.zeros([num_user, 40]).astype(np.int32))
    u_test_gnew_neighs = Tensor(np.zeros([num_user, 20]).astype(np.int32))
    i_test_neighs = Tensor(np.zeros([num_item, 40]).astype(np.int32))
    i_test_gnew_neighs = Tensor(np.zeros([num_item, 20]).astype(np.int32))
    input_data = [users, items, neg_items, u_test_neighs, u_test_gnew_neighs, i_test_neighs, i_test_gnew_neighs]
    file_name = "bgcf"
    export(forward_net, *input_data, file_name=file_name, file_format="MINDIR")
    mindir_file = file_name + ".mindir"
    assert os.path.exists(mindir_file)
    os.remove(mindir_file)
    export(forward_net, *input_data, file_name=file_name, file_format="AIR")
    air_file = file_name + ".air"
    assert os.path.exists(air_file)
    os.remove(air_file)

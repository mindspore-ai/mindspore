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
BGCF training script.
"""
import time

from mindspore import Tensor
import mindspore.context as context
from mindspore.common import dtype as mstype
from mindspore.train.serialization import save_checkpoint
from mindspore.common import set_seed

from src.bgcf import BGCF
from src.config import parser_args
from src.utils import convert_item_id
from src.callback import TrainBGCF
from src.dataset import load_graph, create_dataset

set_seed(1)

def train():
    """Train"""
    num_user = train_graph.graph_info()["node_num"][0]
    num_item = train_graph.graph_info()["node_num"][1]
    num_pairs = train_graph.graph_info()['edge_num'][0]

    bgcfnet = BGCF([parser.input_dim, num_user, num_item],
                   parser.embedded_dimension,
                   parser.activation,
                   parser.neighbor_dropout,
                   num_user,
                   num_item,
                   parser.input_dim)

    train_net = TrainBGCF(bgcfnet, parser.num_neg, parser.l2, parser.learning_rate,
                          parser.epsilon, parser.dist_reg)
    train_net.set_train(True)

    itr = train_ds.create_dict_iterator(parser.num_epoch, output_numpy=True)
    num_iter = int(num_pairs / parser.batch_pairs)

    for _epoch in range(1, parser.num_epoch + 1):

        epoch_start = time.time()
        iter_num = 1

        for data in itr:

            u_id = Tensor(data["users"], mstype.int32)
            pos_item_id = Tensor(convert_item_id(data["items"], num_user), mstype.int32)
            neg_item_id = Tensor(convert_item_id(data["neg_item_id"], num_user), mstype.int32)
            pos_users = Tensor(data["pos_users"], mstype.int32)
            pos_items = Tensor(convert_item_id(data["pos_items"], num_user), mstype.int32)

            u_group_nodes = Tensor(data["u_group_nodes"], mstype.int32)
            u_neighs = Tensor(convert_item_id(data["u_neighs"], num_user), mstype.int32)
            u_gnew_neighs = Tensor(convert_item_id(data["u_gnew_neighs"], num_user), mstype.int32)

            i_group_nodes = Tensor(convert_item_id(data["i_group_nodes"], num_user), mstype.int32)
            i_neighs = Tensor(data["i_neighs"], mstype.int32)
            i_gnew_neighs = Tensor(data["i_gnew_neighs"], mstype.int32)

            neg_group_nodes = Tensor(convert_item_id(data["neg_group_nodes"], num_user), mstype.int32)
            neg_neighs = Tensor(data["neg_neighs"], mstype.int32)
            neg_gnew_neighs = Tensor(data["neg_gnew_neighs"], mstype.int32)

            train_loss = train_net(u_id,
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

            if iter_num == num_iter:
                print('Epoch', '%03d' % _epoch, 'iter', '%02d' % iter_num,
                      'loss',
                      '{}, cost:{:.4f}'.format(train_loss, time.time() - epoch_start))
            iter_num += 1

        if _epoch % parser.eval_interval == 0:
            save_checkpoint(bgcfnet, parser.ckptpath + "/bgcf_epoch{}.ckpt".format(_epoch))


if __name__ == "__main__":
    parser = parser_args()

    context.set_context(mode=context.GRAPH_MODE,
                        device_target=parser.device_target,
                        save_graphs=False)

    if parser.device_target == "Ascend":
        context.set_context(device_id=int(parser.device))

    train_graph, _, sampled_graph_list = load_graph(parser.datapath)
    train_ds = create_dataset(train_graph, sampled_graph_list, parser.workers, batch_size=parser.batch_pairs,
                              num_samples=parser.raw_neighs, num_bgcn_neigh=parser.gnew_neighs, num_neg=parser.num_neg)

    train()

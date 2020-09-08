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
import os
import datetime

from mindspore import Tensor
import mindspore.context as context
from mindspore.common import dtype as mstype
from mindspore.train.serialization import save_checkpoint, load_checkpoint

from src.bgcf import BGCF
from src.metrics import BGCFEvaluate
from src.config import parser_args
from src.utils import BGCFLogger, convert_item_id
from src.callback import ForwardBGCF, TrainBGCF, TestBGCF
from src.dataset import load_graph, create_dataset, TestGraphDataset


def train_and_eval():
    """Train and eval"""
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

    eval_class = BGCFEvaluate(parser, train_graph, test_graph, parser.Ks)

    itr = train_ds.create_dict_iterator(parser.num_epoch)
    num_iter = int(num_pairs / parser.batch_pairs)

    for _epoch in range(1, parser.num_epoch + 1):

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
                      '{}'.format(train_loss))
            iter_num += 1

        if _epoch % parser.eval_interval == 0:
            if os.path.exists("ckpts/bgcf.ckpt"):
                os.remove("ckpts/bgcf.ckpt")
            save_checkpoint(bgcfnet, "ckpts/bgcf.ckpt")

            bgcfnet_test = BGCF([parser.input_dim, num_user, num_item],
                                parser.embedded_dimension,
                                parser.activation,
                                [0.0, 0.0, 0.0],
                                num_user,
                                num_item,
                                parser.input_dim)

            load_checkpoint("ckpts/bgcf.ckpt", net=bgcfnet_test)

            forward_net = ForwardBGCF(bgcfnet_test)
            user_reps, item_reps = TestBGCF(forward_net, num_user, num_item, parser.input_dim, test_graph_dataset)

            test_recall_bgcf, test_ndcg_bgcf, \
            test_sedp, test_nov = eval_class.eval_with_rep(user_reps, item_reps, parser)

            if parser.log_name:
                log.write(
                    'epoch:%03d,      recall_@10:%.5f,     recall_@20:%.5f,     ndcg_@10:%.5f,    ndcg_@20:%.5f,   '
                    'sedp_@10:%.5f,     sedp_@20:%.5f,    nov_@10:%.5f,    nov_@20:%.5f\n' % (_epoch,
                                                                                              test_recall_bgcf[1],
                                                                                              test_recall_bgcf[2],
                                                                                              test_ndcg_bgcf[1],
                                                                                              test_ndcg_bgcf[2],
                                                                                              test_sedp[0],
                                                                                              test_sedp[1],
                                                                                              test_nov[1],
                                                                                              test_nov[2]))
            else:
                print('epoch:%03d,      recall_@10:%.5f,     recall_@20:%.5f,     ndcg_@10:%.5f,    ndcg_@20:%.5f,   '
                      'sedp_@10:%.5f,     sedp_@20:%.5f,    nov_@10:%.5f,    nov_@20:%.5f\n' % (_epoch,
                                                                                                test_recall_bgcf[1],
                                                                                                test_recall_bgcf[2],
                                                                                                test_ndcg_bgcf[1],
                                                                                                test_ndcg_bgcf[2],
                                                                                                test_sedp[0],
                                                                                                test_sedp[1],
                                                                                                test_nov[1],
                                                                                                test_nov[2]))


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend",
                        save_graphs=False)

    parser = parser_args()

    train_graph, test_graph, sampled_graph_list = load_graph(parser.datapath)
    train_ds = create_dataset(train_graph, sampled_graph_list, batch_size=parser.batch_pairs)
    test_graph_dataset = TestGraphDataset(train_graph, sampled_graph_list, num_samples=parser.raw_neighs,
                                          num_bgcn_neigh=parser.gnew_neighs,
                                          num_neg=parser.num_neg)

    if parser.log_name:
        now = datetime.datetime.now().strftime("%b_%d_%H_%M_%S")
        name = "bgcf" + '-' + parser.log_name + '-' + parser.dataset
        log_save_path = './log-files/' + name + '/' + now
        log = BGCFLogger(logname=name, now=now, foldername='log-files', copy=False)
        log.open(log_save_path + '/log.train.txt', mode='a')
        for arg in vars(parser):
            log.write(arg + '=' + str(getattr(parser, arg)) + '\n')
    else:
        for arg in vars(parser):
            print(arg + '=' + str(getattr(parser, arg)))

    train_and_eval()

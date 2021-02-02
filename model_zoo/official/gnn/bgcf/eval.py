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
BGCF evaluation script.
"""
import datetime

import mindspore.context as context
from mindspore.train.serialization import load_checkpoint
from mindspore.common import set_seed

from src.bgcf import BGCF
from src.utils import BGCFLogger
from src.config import parser_args
from src.metrics import BGCFEvaluate
from src.callback import ForwardBGCF, TestBGCF
from src.dataset import TestGraphDataset, load_graph

set_seed(1)

def evaluation():
    """evaluation"""
    num_user = train_graph.graph_info()["node_num"][0]
    num_item = train_graph.graph_info()["node_num"][1]

    eval_class = BGCFEvaluate(parser, train_graph, test_graph, parser.Ks)
    for _epoch in range(parser.eval_interval, parser.num_epoch+1, parser.eval_interval) \
                  if parser.device_target == "Ascend" else range(parser.num_epoch, parser.num_epoch+1):
        bgcfnet_test = BGCF([parser.input_dim, num_user, num_item],
                            parser.embedded_dimension,
                            parser.activation,
                            [0.0, 0.0, 0.0],
                            num_user,
                            num_item,
                            parser.input_dim)

        load_checkpoint(parser.ckptpath + "/bgcf_epoch{}.ckpt".format(_epoch), net=bgcfnet_test)

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
    parser = parser_args()
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=parser.device_target,
                        save_graphs=False)
    if parser.device_target == "Ascend":
        context.set_context(device_id=int(parser.device))

    train_graph, test_graph, sampled_graph_list = load_graph(parser.datapath)
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

    evaluation()

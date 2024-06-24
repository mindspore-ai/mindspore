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

import time
import pytest
import numpy as np
from mindspore import context
from mindspore import Tensor
from tests.models.official.gnn.gcn.src.gcn import GCN
from tests.models.official.gnn.gcn.src.metrics import LossAccuracyWrapper, TrainNetWrapper
from tests.models.official.gnn.gcn.src.config import ConfigGCN
from tests.models.official.gnn.gcn.src.dataset import get_adj_features_labels, get_mask

from tests.mark_utils import arg_mark

DATA_DIR = '/home/workspace/mindspore_dataset/cora/cora_mr/cora_mr'
TRAIN_NODE_NUM = 140
EVAL_NODE_NUM = 500
TEST_NODE_NUM = 1000
SEED = 20


@pytest.mark.skip(reason="because GraphData API was deleted in dataset")
@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_gcn():
    print("test_gcn begin")
    np.random.seed(SEED)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    config = ConfigGCN()
    config.dropout = 0.0
    adj, feature, label_onehot, _ = get_adj_features_labels(DATA_DIR)

    nodes_num = label_onehot.shape[0]
    train_mask = get_mask(nodes_num, 0, TRAIN_NODE_NUM)
    eval_mask = get_mask(nodes_num, TRAIN_NODE_NUM, TRAIN_NODE_NUM + EVAL_NODE_NUM)
    test_mask = get_mask(nodes_num, nodes_num - TEST_NODE_NUM, nodes_num)

    class_num = label_onehot.shape[1]
    input_dim = feature.shape[1]
    gcn_net = GCN(config, input_dim, class_num)
    gcn_net.add_flags_recursive(fp16=True)

    adj = Tensor(adj)
    feature = Tensor(feature)

    eval_net = LossAccuracyWrapper(gcn_net, label_onehot, eval_mask, config.weight_decay)
    test_net = LossAccuracyWrapper(gcn_net, label_onehot, test_mask, config.weight_decay)
    train_net = TrainNetWrapper(gcn_net, label_onehot, train_mask, config)

    loss_list = []
    best_acc = 0.
    for epoch in range(config.epochs):
        t = time.time()

        train_net.set_train()
        train_result = train_net(adj, feature)
        train_loss = train_result[0].asnumpy()
        train_accuracy = train_result[1].asnumpy()

        eval_net.set_train(False)
        eval_result = eval_net(adj, feature)
        eval_loss = eval_result[0].asnumpy()
        eval_accuracy = eval_result[1].asnumpy()

        loss_list.append(eval_loss)
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss),
              "train_acc=", "{:.5f}".format(train_accuracy), "val_loss=", "{:.5f}".format(eval_loss),
              "val_acc=", "{:.5f}".format(eval_accuracy), "time=", "{:.5f}".format(time.time() - t))
        if epoch % 5 == 0:
            test_net.set_train(False)
            test_result = test_net(adj, feature)
            test_loss = test_result[0].asnumpy()
            test_accuracy = test_result[1].asnumpy()
            print("Test set results:", "loss=", "{:.5f}".format(test_loss), "accuracy=", "{:.5f}".format(test_accuracy))
            best_acc = test_accuracy if test_accuracy > best_acc else best_acc

        if epoch > config.early_stopping and loss_list[-1] > np.mean(loss_list[-(config.early_stopping+1):-1]):
            print("Early stopping...")
            break

    assert best_acc > 0.812

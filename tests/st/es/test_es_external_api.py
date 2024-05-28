# Copyright 2024 Huawei Technologies Co., Ltd
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

import os
import numpy as np
import mindspore as ms
from  mindspore import nn, ops, Tensor
from mindspore.nn.layer.embedding_service import EmbeddingService
from mindspore.nn.layer.embedding_service_layer import EsEmbeddingLookup
from mindspore.communication import init, release, get_rank
from mindspore import context


class Net(nn.Cell):
    """
    EsNet
    """
    def __init__(self, embedding_dim, max_feature_count, table_id_dict=None, es_initializer=None,
                 es_counter_filter=None):
        super(Net, self).__init__()
        self.table_id = table_id_dict["test"]
        self.embedding = EsEmbeddingLookup(self.table_id, es_initializer[self.table_id], embedding_dim=embedding_dim,
                                           max_key_num=max_feature_count, optimizer_mode="adam",
                                           es_filter=es_counter_filter[self.table_id])
        self.w = ms.Parameter(Tensor([1.5], ms.float32), name="w", requires_grad=True)

    def construct(self, keys, actual_keys_input=None, unique_indices=None):
        if (actual_keys_input is not None) and (unique_indices is not None):
            es_out = self.embedding(keys, actual_keys_input, unique_indices)
        else:
            es_out = self.embedding(keys)
        output = es_out * self.w
        return output


class NetworkWithLoss(nn.Cell):
    """
    NetworkWithLoss
    """
    def __init__(self, network, loss):
        super(NetworkWithLoss, self).__init__()
        self.network = network
        self.loss_fn = loss

    def construct(self, x, label):
        logits = self.network(x)
        loss = self.loss_fn(logits, label)
        return loss


def train():
    """
    train net.
    """
    init()
    vocab_size = 1000
    embedding_dim = 12
    feature_length = 16
    context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

    es = EmbeddingService()
    filter_option = es.counter_filter(filter_freq=2, default_value=10.0)
    ev_option = es.embedding_variable_option(filter_option=filter_option)

    table_id_dict, es_initializer, es_counter_filter = es.embedding_init("test", init_vocabulary_size=vocab_size,
                                                                         embedding_dim=embedding_dim,
                                                                         max_feature_count=feature_length,
                                                                         optimizer="adam", ev_option=ev_option,
                                                                         mode="train")
    print("Succ do embedding_init: ", table_id_dict, es_initializer, es_counter_filter, flush=True)


    net = Net(embedding_dim, feature_length, table_id_dict, es_initializer, es_counter_filter)
    loss_fn = ops.SigmoidCrossEntropyWithLogits()
    optimizer = nn.Adam(params=net.trainable_params(), learning_rate=1e-3)
    net_with_loss = NetworkWithLoss(net, loss_fn)
    train_network = nn.TrainOneStepCell(net_with_loss, optimizer=optimizer)
    train_network.set_train()

    data = Tensor(np.array(np.ones((2, 8)), dtype=np.float32))
    label = Tensor(np.array(np.ones((2, 8, 12)), dtype=np.float32))

    loss = train_network(data, label)
    print("Succ do train, loss is: ", loss, flush=True)

    rank = get_rank()
    print("Succ get rank, rank is: ", rank, flush=True)
    if rank == 0:
        save_embedding_path = os.path.join(os.getcwd(), "embedding")
        save_ckpt_path = os.path.join(os.getcwd(), "ckpt")
        print("After get path is: ", save_embedding_path, save_ckpt_path, flush=True)
        es.embedding_table_export(save_embedding_path)
        print("Succ do export embedding.", flush=True)
        es.embedding_ckpt_export(save_ckpt_path)
        print("Succ do export ckpt.", flush=True)

    es.embedding_ckpt_import(save_ckpt_path)
    print("Succ do import embedding.", flush=True)

    table_id_dict, es_initializer, es_counter_filter = es.embedding_init("test", init_vocabulary_size=vocab_size,
                                                                         embedding_dim=embedding_dim,
                                                                         max_feature_count=feature_length,
                                                                         optimizer="adam", ev_option=ev_option,
                                                                         mode="predict")
    es.embedding_table_import(save_embedding_path)
    print("Succ do import ckpt.", flush=True)
    release()


if __name__ == "__main__":
    train()

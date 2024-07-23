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
import sys
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
                 es_counter_filter=None, es_padding_key=None, es_completion_key=None):
        super(Net, self).__init__()
        self.table_id = table_id_dict["test"]
        self.embedding_dim = [embedding_dim]
        self.embedding = EsEmbeddingLookup(self.table_id, es_initializer[self.table_id],
                                           embedding_dim=self.embedding_dim,
                                           max_key_num=max_feature_count, optimizer_mode="adam",
                                           optimizer_params=[0.0, 0.0],
                                           es_filter=es_counter_filter[self.table_id],
                                           es_padding_key=es_padding_key[self.table_id],
                                           es_completion_key=es_completion_key[self.table_id])
        self.w = ms.Parameter(Tensor([1.5], ms.float32), name="w", requires_grad=True)

    def construct(self, keys, actual_keys_input=None, unique_indices=None, key_count=None):
        if (actual_keys_input is not None) and (unique_indices is not None):
            es_out = self.embedding(keys, actual_keys_input, unique_indices, key_count)
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
    vocab_size = 1000
    embedding_dim = 12
    feature_length = 16
    context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O2"})
    init()

    es = EmbeddingService()
    filter_option = es.counter_filter(filter_freq=2, default_value=10.0)
    evict_option = es.evict_option(steps_to_live=1)
    padding_option = es.padding_param(padding_key=1, mask=True, mask_zero=False)
    completion_option = es.completion_key(completion_key=3, mask=True)
    ev_option = es.embedding_variable_option(filter_option=filter_option, padding_option=padding_option,
                                             evict_option=evict_option,
                                             completion_option=completion_option)

    es_out = es.embedding_init("test", init_vocabulary_size=vocab_size, embedding_dim=embedding_dim,
                               max_feature_count=feature_length, optimizer="adam", ev_option=ev_option,
                               mode="train")
    table_id_dict = es_out.table_id_dict
    es_initializer = es_out.es_initializer
    es_counter_filter = es_out.es_counter_filter
    es_padding_keys = es_out.es_padding_keys
    es_completion_keys = es_out.es_completion_keys

    if "test" not in table_id_dict:
        raise ValueError("test table should be in table_id_dict")
    if len(es_initializer) != 1 or len(es_counter_filter) != 1 or len(es_padding_keys) != 1 \
            or len(es_completion_keys) != 1:
        raise ValueError("es out len should be 1")

    print("Succ do embedding_init: ", table_id_dict, es_initializer, es_counter_filter,
          es_padding_keys, es_completion_keys, flush=True)

    net = Net(embedding_dim, feature_length, table_id_dict, es_initializer, es_counter_filter,
              es_padding_keys, es_completion_keys)
    loss_fn = ops.SigmoidCrossEntropyWithLogits()
    optimizer = nn.Adam(params=net.trainable_params(), learning_rate=1e-3)
    net_with_loss = NetworkWithLoss(net, loss_fn)
    train_network = nn.TrainOneStepCell(net_with_loss, optimizer=optimizer)
    train_network.set_train()
    data = Tensor(np.array(np.ones((2, 8)), dtype=np.float32))
    label = Tensor(np.array(np.ones((2, 8, 12)), dtype=np.float32))
    for i in range(10):
        loss = train_network(data, label)
        assert loss.shape == (2, 8, 12), 'loss shape should be (2, 8, 12)'
        eval_out = net(data)
        assert eval_out.shape == (2, 8, 12), 'eval_out shape should be (2, 8, 12)'
        if np.any(np.isnan(loss.asnumpy())) or np.any(np.isnan(eval_out.asnumpy())):
            raise ValueError(f"loss or eval_out should not be nan on epoch: {i}!")

    rank = get_rank()
    if rank == 0:
        save_embedding_path = os.path.join(os.getcwd(), "embedding")
        save_ckpt_path = os.path.join(os.getcwd(), "ckpt")
        print("After get path is: ", save_embedding_path, save_ckpt_path, flush=True)
        es.embedding_table_export(save_embedding_path)
        es.embedding_ckpt_export(save_ckpt_path)
        if os.path.exists("embedding_0.test.bin") and os.path.exists("ckpt_0.test.bin") and \
                os.path.exists("ckpt_0.test.meta"):
            print("Succ do export embedding and ckpt.", flush=True)
        else:
            raise ValueError("Fail do export!!!!")

        es.embedding_ckpt_import(save_ckpt_path)
        print("Succ do import embedding.", flush=True)

    release()


if __name__ == "__main__":
    try:
        train()
        sys.exit(0)
    except Exception as e:
        raise e

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
import sys
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from mindspore import Tensor, context
from mindspore import ops, nn
from mindspore.communication import init, release
from mindspore.ops.operations.manually_defined import ops_def

init()

table_id = 0
embedding_dim = 12
value_total_len = 38
max_key_num = 20480


# Init
class InitNet(nn.Cell):
    """
    Init ES hash map.
    """

    def __init__(self, embedding_dim_, value_total_len_):
        super(InitNet, self).__init__()
        self.ps_num_tensor = Tensor(1, ms.int32)
        self.ps_ids_tensor = Tensor([0], ms.int32)
        self.table_id_tensor = Tensor(table_id, ms.int32)
        self.es_op1 = ops_def.init_partition_map
        self.es_op2 = ops_def.init_embedding_hashmap
        self.depend = ops.Depend()
        self.embedding_dim = embedding_dim_
        self.value_total_len = value_total_len_

    def construct(self):
        es_op1 = self.es_op1(
            self.ps_num_tensor,
            self.ps_ids_tensor,
            _embedding_dim=self.embedding_dim,
            _max_key_num=max_key_num,
        )
        z = self.depend(self.table_id_tensor, es_op1)
        es_op2 = self.es_op2(
            z,
            value_total_len=self.value_total_len,
            embedding_dim=self.embedding_dim,
            _table_id=table_id,
            bucket_size=100,
            seed=1024,
            seed2=1024,
        )
        return es_op2


def init_es_net_func(embedding_dim_, value_total_len_):
    """
    int es.
    """
    init_net = InitNet(embedding_dim_, value_total_len_)
    _ = init_net()


# FindAndInit
def embedding_table_find_and_init_forward_func(table_id_,
                                               keys, max_grad_norm, parameter):
    """
    embedding_table_find_and_init.
    """
    y = ops_def.embedding_table_find_and_init(
        table_id_,
        keys,
        max_grad_norm,
        parameter,
        embedding_dim,
        value_total_len,
        _table_id=table_id,
        _max_key_num=max_key_num,
    )
    return y


class EsNet(nn.Cell):
    """
    EsNet
    """

    def __init__(self):
        super(EsNet, self).__init__()
        self.cast = ops.cast
        self.table_id = Tensor(table_id, ms.int64)
        self.max_grad_norm = Tensor([1.0], ms.float32)
        self.w = ms.Parameter(Tensor([2.5], ms.float32), name="w", requires_grad=True)
        self.b = ms.Parameter(Tensor([0], ms.float32), name="b", requires_grad=True)

    def construct(self, keys):
        table_id_ = self.cast(self.table_id, ms.int32)
        y = embedding_table_find_and_init_forward_func(
            table_id_, keys, self.max_grad_norm, self.b
        )
        z = y * self.w
        return z


def source():
    """
    Dataset.
    """
    nums = 30
    keys = np.concatenate((np.zeros(20454), np.arange(26)), 0)
    data = [keys for i in range(nums)]
    label = [np.ones((20480,)) for i in range(nums)]
    for i in range(nums):
        yield data[i].astype(np.int64), label[i].astype(np.int32)


dataloader = ds.GeneratorDataset(source, column_names=["key", "value"])


def train():
    """
    train net.
    """
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend", jit_config={"jit_level": 'O0'})
    init_es_net_func(embedding_dim, value_total_len)

    net = EsNet()

    net_loss = nn.CrossEntropyLoss()
    net_opt = nn.SGD(net.trainable_params(), 1e-2)
    model = ms.train.Model(net, net_loss, net_opt)
    print("============ Start Training ===============")
    for key, value in dataloader:
        print(f"key.shape: {key.shape}")
        print(f"value.shape: {value.shape}")
    model.train(
        epoch=1,
        train_dataset=dataloader,
        callbacks=[ms.train.LossMonitor()],
        dataset_sink_mode=True,
    )


if __name__ == "__main__":
    try:
        train()
        release()
        sys.exit(0)
    except Exception as e:
        raise e

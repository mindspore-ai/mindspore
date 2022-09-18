# Copyright 2022 Huawei Technologies Co., Ltd
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
import copy
import numpy as np
import mindspore
from mindspore.nn import Cell, Flatten, Dense
from mindspore.nn import EmbeddingLookup, SoftmaxCrossEntropyWithLogits
from mindspore.nn import Adam
from mindspore.train import Model
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.train.metrics import Accuracy
from mindspore.common import set_seed
from mindspore.communication.management import get_rank
import mindspore.ops.operations as op


class Net(Cell):
    def __init__(self, in_channels, out_channels, vocab_size, embedding_size,
                 target, sparse, vocab_cache_size):
        super().__init__()
        set_seed(5)
        self.embedding_lookup1 = EmbeddingLookup(vocab_size=vocab_size,
                                                 embedding_size=embedding_size,
                                                 param_init='normal', target=target,
                                                 slice_mode='table_row_slice', sparse=sparse,
                                                 vocab_cache_size=vocab_cache_size)
        self.embedding_lookup2 = EmbeddingLookup(vocab_size=vocab_size,
                                                 embedding_size=embedding_size,
                                                 param_init='normal', target=target,
                                                 slice_mode='table_row_slice', sparse=sparse,
                                                 vocab_cache_size=vocab_cache_size)
        self.embedding_lookup3 = EmbeddingLookup(vocab_size=vocab_size,
                                                 embedding_size=embedding_size,
                                                 param_init='normal', target=target,
                                                 slice_mode='table_row_slice', sparse=sparse,
                                                 vocab_cache_size=vocab_cache_size)
        self.add = op.TensorAdd()
        self.flatten = Flatten()
        self.dense = Dense(in_channels=in_channels, out_channels=out_channels, weight_init='normal',
                           has_bias=False)
        self.type = mindspore.int32
        self.cast = op.Cast()

    def construct(self, x):
        x = self.flatten(x)
        x = self.cast(x, self.type)
        y = self.embedding_lookup1(x)
        z = self.embedding_lookup2(x)
        u = self.embedding_lookup3(x)
        x = self.add(y, z)
        x = self.add(x, u)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.flatten(x)
        return x


class ModelExecutor:
    def __init__(self, dataset, input_shape, in_channels=320, out_channels=3, vocab_size=50,
                 embedding_size=10, epoch_size=2, opt_ps='adam', dense_ps=False, target='DEVICE',
                 sparse=False, vocab_cache_size=0, save_ckpt=False, dtype=np.int32):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.train_dataset = dataset
        self.eval_dataset = copy.deepcopy(dataset)
        self.epoch_size = epoch_size
        self.target = target
        self.sparse = sparse
        self.vocab_cache_size = vocab_cache_size
        self.save_ckpt = save_ckpt

    def run_embedding_cache(self):
        net = Net(self.in_channels, self.out_channels, self.vocab_size, self.embedding_size,
                  self.target, self.sparse, self.vocab_cache_size)
        net.embedding_lookup1.set_param_ps()
        net.embedding_lookup2.set_param_ps()
        net.embedding_lookup3.set_param_ps()
        net.set_train()
        loss = SoftmaxCrossEntropyWithLogits(reduction='mean')
        opt = Adam(params=filter(lambda x: x.requires_grad, net.get_parameters()))

        model = Model(net, loss, opt, metrics={"Accuracy": Accuracy()})
        callback_list = []
        if self.save_ckpt:
            config = CheckpointConfig(save_checkpoint_steps=1, keep_checkpoint_max=1)
            ms_role = os.getenv("MS_ROLE")
            ckpoint_cb_ps = ModelCheckpoint(prefix="CKPT_PS", directory='./ckpt_' + ms_role + str(get_rank()) + '/',
                                            config=config)
            callback_list.append(ckpoint_cb_ps)
        model.train(self.epoch_size, self.train_dataset, callbacks=callback_list, dataset_sink_mode=True)
        acc = model.eval(self.eval_dataset, dataset_sink_mode=True)
        return acc['Accuracy']

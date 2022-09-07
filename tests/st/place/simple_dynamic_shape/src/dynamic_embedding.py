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
import numpy as np
import mindspore
import mindspore.dataset as ds
from mindspore.nn import Cell
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.nn import Adam
from mindspore.nn import EmbeddingLookup
import mindspore.ops.operations as P
from mindspore.common.initializer import TruncatedNormal

VOCAB_SIZE = 320000
EMBEDDING_SIZE = 10
BATCH_SIZE = 32
MINI_BATCH_NUM = 20

if "embedding_size" in os.environ:
    EMBEDDING_SIZE = int(os.environ["embedding_size"])
if "batch_size" in os.environ:
    BATCH_SIZE = int(os.environ["batch_size"])


def create_dataset():
    # Each epoch has mini_batch_num data.
    np_data = np.random.randint(0, VOCAB_SIZE - 1, (MINI_BATCH_NUM, BATCH_SIZE))
    np_label = np.random.randint(0, EMBEDDING_SIZE - 1, (MINI_BATCH_NUM, BATCH_SIZE))
    dataset = ds.NumpySlicesDataset((np_data, np_label), shuffle=False)
    return dataset


def weight_variable():
    """weight initial"""
    return TruncatedNormal(0.02)


class Net(Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.type = mindspore.int32
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.embedding_lookup = EmbeddingLookup(vocab_size=VOCAB_SIZE,
                                                embedding_size=EMBEDDING_SIZE,
                                                param_init=weight_variable(),
                                                sparse=True,
                                                target='DEVICE')
        self.embedding_lookup.forward_unique = True
        self.embedding_lookup.gatherv2.add_prim_attr('rank_id', 0)
        self.embedding_lookup.gatherv2.add_prim_attr('ms_role', 'MS_PSERVER')

    def construct(self, input_x):
        output = self.reshape(input_x, (BATCH_SIZE, -1))
        output = self.cast(output, self.type)
        output = self.embedding_lookup(output)
        output = self.reshape(output, (BATCH_SIZE, -1))
        return output


def get_optimizer(net):
    adam_optimizer = Adam(filter(lambda x: x.requires_grad, net.get_parameters()))
    return adam_optimizer


def get_loss():
    return SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')


def get_dataset():
    return create_dataset()

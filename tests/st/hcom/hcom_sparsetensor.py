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
import os
import numpy as np
from mindspore.communication.management import get_rank
from mindspore import Tensor
from mindspore import Parameter
from mindspore import context
from mindspore.ops import operations as P
import mindspore.nn as nn
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.communication.management import init
from mindspore.communication.management import get_group_size


class FakeDataInitMode:
    RandomInit = 0
    OnesInit = 1
    UniqueInit = 2
    ZerosInit = 3

class FakeData:
    def __init__(self, size=1024, batch_size=32, image_size=(3, 224, 224), num_class=10,
                 random_offset=0, use_parallel=False, fakedata_mode=FakeDataInitMode.RandomInit):

        self.size = size
        self.rank_batch_size = batch_size
        self.total_batch_size = self.rank_batch_size
        self.random_offset = random_offset
        self.image_size = image_size
        self.num_class = num_class
        self.rank_size = 1
        self.rank_id = 0
        self.batch_index = 0
        self.image_data_type = np.float32
        self.label_data_type = np.float32
        self.is_onehot = True
        self.fakedata_mode = fakedata_mode

        if use_parallel:
            if 'CONTEXT_DEVICE_TARGET' in os.environ and os.environ['CONTEXT_DEVICE_TARGET'] == 'GPU':
                init(backend_name='nccl')
            else:
                init(backend_name='hccl')
            self.rank_size = get_group_size()
            self.rank_id = get_rank()
        self.total_batch_size = self.rank_batch_size * self.rank_size
        assert self.size % self.total_batch_size == 0
        self.total_batch_data_size = (self.rank_size, self.rank_batch_size) + image_size

    def get_dataset_size(self):
        return int(self.size / self.total_batch_size)

    def get_reeat_count(self):
        return 1

    def set_image_data_type(self, data_type):
        self.image_data_type = data_type

    def set_label_data_type(self, data_type):
        self.label_data_type = data_type

    def set_label_onehot(self, is_onehot=True):
        self.is_onehot = is_onehot

    def create_tuple_iterator(self, num_epochs=-1, do_copy=False):
        return self

    def __getitem__(self, batch_index):
        if batch_index * self.total_batch_size >= len(self):
            raise IndexError("{} index out of range".format(self.__class__.__name__))
        rng_state = np.random.get_state()
        np.random.seed(batch_index + self.random_offset)
        if self.fakedata_mode == FakeDataInitMode.OnesInit:
            img = np.ones(self.total_batch_data_size)
        elif self.fakedata_mode == FakeDataInitMode.ZerosInit:
            img = np.zeros(self.total_batch_data_size)
        elif self.fakedata_mode == FakeDataInitMode.UniqueInit:
            total_size = 1
            for i in self.total_batch_data_size:
                total_size = total_size* i
            img = np.reshape(np.arange(total_size)*0.0001, self.total_batch_data_size)
        else:
            img = np.random.randn(*self.total_batch_data_size)
        target = np.random.randint(0, self.num_class, size=(self.rank_size, self.rank_batch_size))
        np.random.set_state(rng_state)
        img = img[self.rank_id]
        target = target[self.rank_id]
        img_ret = img.astype(self.image_data_type)
        target_ret = target.astype(self.label_data_type)
        if self.is_onehot:
            target_onehot = np.zeros(shape=(self.rank_batch_size, self.num_class))
            target_onehot[np.arange(self.rank_batch_size), target] = 1
            target_ret = target_onehot.astype(self.label_data_type)
        return Tensor(img_ret), Tensor(target_ret)

    def __len__(self):
        return self.size

    def __iter__(self):
        self.batch_index = 0
        return self

    def reset(self):
        self.batch_index = 0

    def __next__(self):
        if self.batch_index * self.total_batch_size < len(self):
            data = self[self.batch_index]
            self.batch_index += 1
            return data
        raise StopIteration


class NetWithSparseGatherV2(nn.Cell):
    def __init__(self, strategy=None, sparse=True):
        super(NetWithSparseGatherV2, self).__init__()
        self.axis = 0
        self.sparse = sparse
        if sparse:
            self.weight = Parameter(Tensor(np.ones([8, 8]).astype(np.float32)), name="weight")
            self.gather = P.SparseGatherV2()
        else:
            self.weight = Parameter(Tensor(np.ones([8, 8]).astype(np.float32)), name="weight")
            self.gather = P.Gather()
        if strategy is not None:
            self.gather.shard(strategy)

    def construct(self, indices):
        x = self.gather(self.weight, indices, self.axis)
        return x

    def train_mindspore_impl(self, indices, epoch, batch_size, use_parallel=True):
        ds = FakeData(size=8, batch_size=batch_size, num_class=8, image_size=(), use_parallel=use_parallel)
        ds.set_image_data_type(np.int32)
        net = self
        net.set_train()
        loss = nn.SoftmaxCrossEntropyWithLogits()
        optimizer = nn.Adam(net.trainable_params())
        optimizer.target = "CPU"
        model = Model(net, loss, optimizer)
        for _ in range(epoch):
            model.train(1, ds, dataset_sink_mode=False)
        output = net(indices)
        return output


def test_allreduce_sparsegatherv2_adam_auto_parallel():
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    init(backend_name='hccl')
    context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, device_num=8, gradients_mean=True)
    indices = Tensor(np.array([0, 1, 2, 3, 4, 5, 6, 7]).astype(np.int32))
    epoch = 3
    batch_size = 1
    context.set_context(enable_sparse=True)
    net = NetWithSparseGatherV2(sparse=True)
    output_sparse = net.train_mindspore_impl(indices, epoch, batch_size)
    net = NetWithSparseGatherV2(sparse=False)
    output = net.train_mindspore_impl(indices, epoch, batch_size)
    assert np.allclose(output.asnumpy(), output_sparse.asnumpy(), 0.001, 0.001)

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

import mindspore.ops.operations as P
from mindspore.nn import Cell
from mindspore.nn import Adam
from mindspore.nn import MultiFieldEmbeddingLookup as embedding
from mindspore import Tensor
from mindspore import context
from mindspore.train import Model
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.serialization import load_checkpoint
from mindspore.train.serialization import load_param_into_net
from mindspore.communication.management import init
from mindspore.communication.management import release
from mindspore.communication.management import get_rank
from mindspore.communication.management import get_group_size
from mindspore.context import ParallelMode


context.set_context(mode=context.GRAPH_MODE, device_target='GPU')


def _count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_me) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
            format(data_expected[greater], data_me[greater], error[greater])


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if np.any(np.isnan(data_expected)):
        assert np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert True

def clean_all_ckpt_files(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.ckpt') or file_name.endswith('.meta'):
                os.remove(os.path.join(folder_path, file_name))


def find_newest_ckpt_file(folder_path):
    ckpt_files = map(lambda f: os.path.join(folder_path, f),
                     filter(lambda f: f.endswith('.ckpt'),
                            os.listdir(folder_path)))
    return max(ckpt_files, key=os.path.getctime)


class FakeDataInitMode:
    RandomInit = 0
    OnesInit = 1
    UniqueInit = 2
    ZerosInit = 3




class FakeData:
    def __init__(self, size=1024, batch_size=32, image_size=(3, 224, 224),
                 num_classes=10, random_offset=0, use_parallel=False,
                 fakedata_mode=FakeDataInitMode.RandomInit):
        self.size = size
        self.rank_batch_size = batch_size
        self.total_batch_size = self.rank_batch_size
        self.random_offset = random_offset
        self.image_size = image_size
        self.num_classes = num_classes
        self.rank_size = 1
        self.rank_id = 0
        self.batch_index = 0
        self.image_data_type = np.float32
        self.label_data_type = np.float32
        self.is_onehot = True
        self.fakedata_mode = fakedata_mode

        if use_parallel is True:
            init(backend_name='nccl')
            self.rank_size = get_group_size()
            self.rank_id = get_rank()

        self.total_batch_size = self.rank_batch_size * self.rank_size

        assert (self.size % self.total_batch_size) == 0

        self.total_batch_data_size = (self.rank_size, self.rank_batch_size) + image_size

    def get_dataset_size(self):
        return int(self.size / self.total_batch_size)

    def get_repeat_count(self):
        return 1

    def set_image_data_type(self, data_type):
        self.image_data_type = data_type

    def set_label_data_type(self, data_type):
        self.label_data_type = data_type

    def set_label_onehot(self, is_onehot=True):
        self.is_onehot = is_onehot

    def create_tuple_iterator(self, num_epochs=-1, do_copy=True):
        _ = num_epochs
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
                total_size = total_size * i
            img = np.reshape(np.arange(total_size) * 0.0001, self.total_batch_data_size)
        else:
            img = np.random.randn(*self.total_batch_data_size)
        target = np.random.randint(0, self.num_classes, size=(self.rank_size, self.rank_batch_size))
        np.random.set_state(rng_state)
        img = img[self.rank_id]
        target = target[self.rank_id]
        img_ret = img.astype(self.image_data_type)
        target_ret = target.astype(self.label_data_type)
        if self.is_onehot:
            target_onehot = np.zeros(shape=(self.rank_batch_size, self.num_classes))
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



class MultiHotNet(Cell):
    def __init__(self, vocab_size, embedding_size, field_size,
                 param_init, target, slice_mode, sparse, operator, indices, field_ids):
        super().__init__()
        self.embedding = embedding(vocab_size=vocab_size,
                                   embedding_size=embedding_size, field_size=field_size,
                                   param_init=param_init, target=target, slice_mode=slice_mode,
                                   sparse=sparse, operator=operator)
        self.relu = P.ReLU()
        self.indices = Tensor(indices)
        self.field_ids = Tensor(field_ids)
        if slice_mode == "table_column_slice":
            self.relu.shard(((1, 1, 8),))
        elif slice_mode == "table_row_slice":
            self.relu.shard(((8, 1, 1),))
        elif slice_mode == "batch_slice":
            self.relu.shard(((8, 1, 1),))

    def construct(self, values, label):
        x = self.embedding(self.indices, values, self.field_ids)
        output = self.relu(x)
        return output


class ParallelMultiHotFactory:
    def __init__(self, vocab_size, embedding_size, field_size,
                 param_init, target, slice_mode, sparse, operator, indices, field_ids):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.field_size = field_size
        self.param_init = param_init
        self.target = target
        self.slice_mode = slice_mode
        self.sparse = sparse
        self.operator = operator
        self.indices = indices
        self.field_ids = field_ids
        self.global_rank_id = None
        self.opt = None
        self.model = None
        self.standalone_ckpt = None
        self.parallel_ckpt = None
        self.loss_fn = None
        self._init_parallel()
        self._set_parallel_env()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def __del__(self):
        self._release_parallel()

    def _set_parallel_env(self):
        self.global_rank_id = get_rank()

    def _init_parallel(self):
        self._init_parallel_flag = False
        init(backend_name='nccl')
        self._init_parallel_flag = True

    def _release_parallel(self):
        release()

    def _model_train_and_save_ckpt(self, net, dataset, epoch):
        self.opt = Adam(params=net.get_parameters())
        if self.target == 'CPU':
            self.opt.target = self.target
        if self.sparse:
            context.set_context(enable_sparse=True)
        self.model = Model(network=net,
                           loss_fn=self.loss_fn,
                           optimizer=self.opt)
        ckpt_config = CheckpointConfig(keep_checkpoint_max=1)
        ckpt_path = './rank_{}_ckpt'.format(self.global_rank_id)
        ckpt_callback = ModelCheckpoint(prefix='parallel', directory=ckpt_path,
                                        config=ckpt_config)
        clean_all_ckpt_files(ckpt_path)
        self.model.train(epoch=epoch,
                         train_dataset=dataset,
                         callbacks=[ckpt_callback],
                         dataset_sink_mode=False)
        newest_ckpt_file = find_newest_ckpt_file(ckpt_path)
        return load_checkpoint(newest_ckpt_file)

    def mindspore_auto_parallel_impl(self, dataset, epoch, device_num):
        context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL,
                                          device_num=device_num)
        parallel_mode_net = MultiHotNet(vocab_size=self.vocab_size, embedding_size=self.embedding_size,
                                        field_size=self.field_size, param_init=self.param_init, target=self.target,
                                        slice_mode=self.slice_mode, sparse=self.sparse, operator=self.operator,
                                        indices=self.indices, field_ids=self.field_ids)
        self.parallel_ckpt = self._model_train_and_save_ckpt(net=parallel_mode_net, epoch=epoch, dataset=dataset)

    def mindspore_standalone_impl(self, epoch, dataset):
        context.set_auto_parallel_context(parallel_mode=ParallelMode.STAND_ALONE)
        stand_alone_net = MultiHotNet(vocab_size=self.vocab_size, embedding_size=self.embedding_size,
                                      field_size=self.field_size, param_init=self.param_init, target=self.target,
                                      slice_mode=self.slice_mode, sparse=self.sparse, operator=self.operator,
                                      indices=self.indices, field_ids=self.field_ids)
        self.standalone_ckpt = self._model_train_and_save_ckpt(net=stand_alone_net,
                                                               epoch=epoch, dataset=dataset)

    def checkpoint_cmp(self, inputs_np, label):
        standalone_net = MultiHotNet(vocab_size=self.vocab_size, embedding_size=self.embedding_size,
                                     field_size=self.field_size, param_init=self.param_init, target=self.target,
                                     slice_mode=self.slice_mode, sparse=self.sparse, operator=self.operator,
                                     indices=self.indices, field_ids=self.field_ids)
        parallel_net = MultiHotNet(vocab_size=self.vocab_size, embedding_size=self.embedding_size,
                                   field_size=self.field_size, param_init=self.param_init, target=self.target,
                                   slice_mode=self.slice_mode, sparse=self.sparse, operator=self.operator,
                                   indices=self.indices, field_ids=self.field_ids)
        load_param_into_net(standalone_net, self.standalone_ckpt)
        load_param_into_net(parallel_net, self.parallel_ckpt)
        standalone_out = standalone_net(Tensor(inputs_np), Tensor(label))
        parallel_out = parallel_net(Tensor(inputs_np), Tensor(label))
        allclose_nparray(standalone_out.asnumpy(), parallel_out.asnumpy(), 0.001, 0.001)

def test_auto_parallel_multifieldembeddinglookup_device_table_column_slice_mean():
    inputs_np = 10 * np.random.randn(64, 64).astype(np.float32)
    label = 10 * np.random.randn(64, 64).astype(np.float32)
    indices = np.random.randint(0, 9, (64, 64), np.int32)
    field_ids = np.random.randint(0, 20, (64, 64), np.int32)
    fact = ParallelMultiHotFactory(vocab_size=32, embedding_size=64, field_size=64, param_init='one', target='DEVICE',
                                   slice_mode='table_column_slice', sparse=False, operator='MEAN',
                                   indices=indices, field_ids=field_ids)

    #stand alone
    standalone_dataset = FakeData(size=64, batch_size=64, image_size=(64,))
    fact.mindspore_standalone_impl(dataset=standalone_dataset, epoch=2)

    #auto parallel
    parallel_dataset = FakeData(size=64, batch_size=8, image_size=(64,), use_parallel=True)
    fact.mindspore_auto_parallel_impl(dataset=parallel_dataset, epoch=2, device_num=8)

    #compare
    fact.checkpoint_cmp(inputs_np=inputs_np, label=label)

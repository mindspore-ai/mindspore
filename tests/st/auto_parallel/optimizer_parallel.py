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

import os
import numpy as np

from mindspore.communication.management import init
from mindspore.communication.management import release
from mindspore.communication.management import get_rank
from mindspore.communication.management import get_group_size
from mindspore.nn import Cell
from mindspore.nn import ReLU
from mindspore.nn import Dense
from mindspore.nn import Flatten
from mindspore.nn import Momentum
import mindspore.ops.operations as P
from mindspore.train.serialization import load_param_into_net
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.serialization import load_checkpoint

from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.train import Model
from mindspore.parallel import set_algo_parameters
from mindspore import Tensor
from mindspore.common.parameter import Parameter
from mindspore import context
from mindspore.context import ParallelMode

context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')

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
            init(backend_name='hccl')
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


class OptimizerSemiAutoAndAutoParallel6Net(Cell):
    def __init__(self, strategy_dict=None):
        super().__init__()
        shared_np = np.full((16, 1, 32, 32), 0.5, dtype=np.float32)
        self.shared_weight = Parameter(Tensor(shared_np), name='shared_weight')
        self.fc1 = Dense(in_channels=1024,
                         out_channels=116,
                         weight_init='ones',
                         bias_init='ones',
                         has_bias=True)
        self.relu = ReLU()
        self.sigmoid = P.Sigmoid()
        self.add1 = P.Add()
        self.add2 = P.Add()
        self.mul1 = P.Mul().add_prim_attr('primitive_target', 'CPU')
        self.mul2 = P.Mul()
        self.mul3 = P.Mul()
        self.flatten = Flatten()

        mul2_weight_np = np.full((16, 116), 1, dtype=np.float32)
        self.mul2_weight = Parameter(Tensor(mul2_weight_np), name='mul2_weight')

        mul3_weight_np = np.full((16, 116), 1, dtype=np.float32)
        self.mul3_weight = Parameter(Tensor(mul3_weight_np), name='mul3_weight')

        if strategy_dict is not None:
            self.add1.shard(strategy_dict['add1'])
            self.mul1.shard(strategy_dict['mul1'])
            self.fc1.matmul.shard(strategy_dict['fc1_matmul'])
            self.fc1.bias_add.shard(strategy_dict['fc1_bias_add'])
            self.mul2.shard(strategy_dict['mul2'])
            self.mul3.shard(strategy_dict['mul3'])

    def construct(self, inputs):
        relu = self.relu(inputs)
        sigmoid = self.sigmoid(inputs)
        add1 = self.add1(relu, self.shared_weight)
        mul = self.mul1(sigmoid, self.shared_weight)
        add2 = self.add2(add1, mul)
        flatten = self.flatten(add2)
        dense = self.fc1(flatten)
        mul2 = self.mul2(dense, self.mul2_weight)
        out = self.mul3(mul2, self.mul3_weight)
        return out


class OptimizerSemiAutoAndAutoParallelFactory:
    def __init__(self, net, strategy_dict=None):
        self.parallel_ckpt = None
        self.optimizer_parallel_ckpt = None
        self.net = net
        self.strategy_dict = strategy_dict
        self.global_rank_id = None
        self._set_parallel_env()
        self._init_parallel()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def __del__(self):
        self._release_parallel()

    def _set_parallel_env(self):
        if 'RANK_ID' in os.environ:
            self.global_rank_id = int(os.environ['RANK_ID'])

    def _init_parallel(self):
        self._init_parallel_flag = False
        init(backend_name='hccl')
        self._init_parallel_flag = True

    def _release_parallel(self):
        if self._init_parallel_flag:
            release()

    def _model_train_and_save_ckpt(self, net, dataset, epoch):
        self.opt = Momentum(learning_rate=0.01, momentum=0.9, params=net.get_parameters())
        self.loss_fn = SoftmaxCrossEntropyWithLogits(reduction='mean')
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

    def mindspore_auto_parallel_impl(self,
                                     dataset,
                                     epoch,
                                     device_num):
        set_algo_parameters(fully_use_devices=False)
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL,
                                          device_num=device_num)
        parallel_mode_net = self.net(self.strategy_dict)
        self.parallel_ckpt = self._model_train_and_save_ckpt(net=parallel_mode_net,
                                                             dataset=dataset, epoch=epoch)
        context.reset_auto_parallel_context()

    def mindspore_optimizer_auto_parallel_impl(self,
                                               dataset,
                                               epoch,
                                               device_num):
        set_algo_parameters(fully_use_devices=False)
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL,
                                          device_num=device_num,
                                          enable_parallel_optimizer=True)
        parallel_mode_net = self.net(self.strategy_dict)
        self.optimizer_parallel_ckpt = self._model_train_and_save_ckpt(net=parallel_mode_net,
                                                                       dataset=dataset, epoch=epoch)
        context.reset_auto_parallel_context()

    def checkpoint_cmp(self, inputs_np):
        optimizer_parallel_net = self.net(self.strategy_dict)
        load_param_into_net(optimizer_parallel_net, self.optimizer_parallel_ckpt)
        optimizer_parallel_out = optimizer_parallel_net(Tensor(inputs_np))

        parallel_net = self.net(self.strategy_dict)
        load_param_into_net(parallel_net, self.parallel_ckpt)
        parallel_out = parallel_net(Tensor(inputs_np))
        allclose_nparray(optimizer_parallel_out.asnumpy(), parallel_out.asnumpy(), 0.001, 0.001)

def test_optimizer_parallel_auto_4p_6_parameter_same_strategy_1_1_2_1_momentum():
    inputs_np = np.random.randn(16, 1, 32, 32).astype(np.float32)
    ds1 = FakeData(size=32,
                   batch_size=4,
                   image_size=(1, 32, 32),
                   use_parallel=True,
                   num_classes=116)

    ds2 = FakeData(size=32,
                   batch_size=4,
                   image_size=(1, 32, 32),
                   use_parallel=True,
                   num_classes=116)
    strategy_dict = {'add1': ((1, 1, 2, 1), (1, 1, 2, 1)),
                     'mul1': ((1, 1, 2, 1), (1, 1, 2, 1)),
                     'fc1_matmul': ((1, 2), (1, 2)),
                     'fc1_bias_add': ((1, 2), (2,)),
                     'mul2': ((1, 2), (1, 2)),
                     'mul3': ((1, 2), (1, 2))}
    fact = OptimizerSemiAutoAndAutoParallelFactory(net=OptimizerSemiAutoAndAutoParallel6Net,
                                                   strategy_dict=strategy_dict)
    fact.mindspore_auto_parallel_impl(dataset=ds1, epoch=2, device_num=4)
    fact.mindspore_optimizer_auto_parallel_impl(dataset=ds2, epoch=2, device_num=4)
    fact.checkpoint_cmp(inputs_np=inputs_np)

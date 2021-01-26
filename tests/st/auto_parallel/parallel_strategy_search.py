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
from mindspore.nn import Conv2d
from mindspore.nn import ReLU
from mindspore.nn import Dense
from mindspore.nn import Softmax
import mindspore.ops.operations as P
from mindspore.train.serialization import load_param_into_net
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.serialization import load_checkpoint
from mindspore.nn import Momentum
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.train import Model
from mindspore.parallel import set_algo_parameters
from mindspore.common.initializer import initializer
from mindspore.common import dtype as mstype
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


class ParallelStrategySearchNet(Cell):
    def __init__(self, in_channel, out_channel, axis, input_shape, mul_size,
                 test_size, prelu_size, transpose_b, matmul_size, num_class):
        super().__init__()
        mul_np = np.full(mul_size, 0.5, dtype=np.float32)
        self.mul_weight = Parameter(Tensor(mul_np), name="mul_weight")
        bias_np = np.full((12,), 7.1, dtype=np.float32)
        self.bias = Parameter(Tensor(bias_np), name="bias")
        prelu_np = np.full(prelu_size, 0.8, dtype=np.float32)
        self.prelu_weight = Parameter(Tensor(prelu_np), name="prelu_weight")
        matmul_np = np.full(matmul_size, 1.1, dtype=np.float32)
        self.matmul_weight = Parameter(Tensor(matmul_np), name="matmul_weight")
        self.mul = P.Mul()
        self.conv = Conv2d(in_channels=in_channel, out_channels=out_channel,
                           kernel_size=5, has_bias=True,
                           weight_init='ones', bias_init='ones',
                           pad_mode='valid')
        self.scalar = 0.5
        self.parameter = Parameter(
            initializer(0.5, test_size, dtype=mstype.float32),
            name='parameter')
        self.tensor = Tensor(np.full(test_size, 0.05, dtype=np.float32))
        self.softmax = Softmax(axis=axis)
        self.relu = ReLU()
        self.relu.relu.add_prim_attr("primitive_target", "CPU")
        self.reshape = P.Reshape()
        self.input_shape = input_shape
        self.equal = P.Equal()
        self.cast = P.Cast()
        self.concat = P.Concat(axis=1)
        self.reduce_sum = P.ReduceSum()
        self.bias_add = P.BiasAdd()
        self.cos = P.Cos()
        self.prelu = P.PReLU()
        self.matmul = P.MatMul(transpose_b=transpose_b)
        self.l2norm = P.L2Normalize(axis=(1 - axis))
        self.tensoradd = P.Add()
        self.strided_slice = P.StridedSlice()
        self.dense = Dense(in_channels=6,
                           out_channels=num_class,
                           weight_init='ones',
                           bias_init='ones',
                           has_bias=True)

    def construct(self, inputs):
        x = self.conv(inputs)
        x = self.softmax(x)
        x = self.relu(x)
        x = self.mul(x, self.mul_weight)
        x = self.reshape(x, self.input_shape)
        y = self.parameter * self.tensor * self.scalar
        z = self.equal(self.parameter, self.scalar)
        z = self.cast(z, mstype.float16)
        z = self.cast(z, mstype.float32)
        x = self.concat((x, y, z))
        x = self.reduce_sum(x, (2, 3))
        x = self.bias_add(x, self.bias)
        y = self.cos(x)
        y = self.prelu(y, self.prelu_weight)
        z = self.matmul(x, self.matmul_weight)
        z = self.l2norm(z)
        x = self.tensoradd(y, z)
        x = self.strided_slice(x, (0, 0), (32, 6), (1, 1))
        x = self.dense(x)
        return x


class ParallelStrategySearchFactory:
    def __init__(self, standalone_mode_net, parallel_mode_net):
        self.standalone_mode_net = standalone_mode_net
        self.parallel_mode_net = parallel_mode_net
        self.parallel_ckpt = None
        self.standalone_ckpt = None
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

    def mindspore_auto_parallel_impl(self, dataset, epoch, device_num, auto_parallel_search_mode="dynamic_programming"):
        parallel_mode_net = self.parallel_mode_net
        set_algo_parameters(fully_use_devices=False)
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL,
                                          device_num=device_num,
                                          auto_parallel_search_mode=auto_parallel_search_mode)
        self.parallel_ckpt = self._model_train_and_save_ckpt(net=parallel_mode_net,
                                                             dataset=dataset, epoch=epoch)
        context.reset_auto_parallel_context()

    def mindspore_standalone_impl(self, dataset, epoch):
        standalone_mode_net = self.standalone_mode_net
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.STAND_ALONE)
        self.standalone_ckpt = self._model_train_and_save_ckpt(net=standalone_mode_net,
                                                               dataset=dataset, epoch=epoch)
        context.reset_auto_parallel_context()

    def checkpoint_cmp(self, inputs_np):
        standalone_net = self.standalone_mode_net
        load_param_into_net(standalone_net, self.standalone_ckpt)
        standalone_out = standalone_net(Tensor(inputs_np))

        parallel_net = self.standalone_mode_net
        load_param_into_net(parallel_net, self.parallel_ckpt)
        parallel_out = parallel_net(Tensor(inputs_np))
        allclose_nparray(standalone_out.asnumpy(), parallel_out.asnumpy(),
                         0.001, 0.001)


def test_auto_parallel_strategy_search_axis_1_basic():
    inputs_np = np.random.randn(32, 3, 224, 224).astype(np.float32)
    standalone_mode_net = ParallelStrategySearchNet(in_channel=3,
                                                    out_channel=8, axis=1, input_shape=(32, 4, 110, -1),
                                                    mul_size=(32, 1, 220, 220), test_size=(32, 4, 110, 880),
                                                    prelu_size=(1,), transpose_b=True, matmul_size=(1, 12),
                                                    num_class=12)
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL)
    parallel_mode_net = ParallelStrategySearchNet(in_channel=3,
                                                  out_channel=8, axis=1, input_shape=(32, 4, 110, -1),
                                                  mul_size=(32, 1, 220, 220), test_size=(32, 4, 110, 880),
                                                  prelu_size=(1,), transpose_b=True, matmul_size=(1, 12),
                                                  num_class=12)
    parallel_mode_net.cos.shard(((2, 4),))
    parallel_mode_net.matmul.shard(((1, 2), (1, 2)))
    standalone_dataset = FakeData(size=128, batch_size=32,
                                  image_size=(3, 224, 224), num_classes=12)
    fact = ParallelStrategySearchFactory(standalone_mode_net=standalone_mode_net,
                                         parallel_mode_net=parallel_mode_net)
    fact.mindspore_standalone_impl(dataset=standalone_dataset, epoch=2)
    parallel_dataset = FakeData(size=128, batch_size=4,
                                image_size=(3, 224, 224), use_parallel=True,
                                num_classes=12)
    fact.mindspore_auto_parallel_impl(dataset=parallel_dataset,
                                      epoch=2, device_num=8)
    fact.checkpoint_cmp(inputs_np=inputs_np)


def test_auto_parallel_recursive_strategy_search_axis_1_basic():
    inputs_np = np.random.randn(32, 3, 224, 224).astype(np.float32)
    standalone_mode_net = ParallelStrategySearchNet(in_channel=3,
                                                    out_channel=8, axis=1, input_shape=(32, 4, 110, -1),
                                                    mul_size=(32, 1, 220, 220), test_size=(32, 4, 110, 880),
                                                    prelu_size=(1,), transpose_b=True, matmul_size=(1, 12),
                                                    num_class=12)
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL)
    parallel_mode_net = ParallelStrategySearchNet(in_channel=3,
                                                  out_channel=8, axis=1, input_shape=(32, 4, 110, -1),
                                                  mul_size=(32, 1, 220, 220), test_size=(32, 4, 110, 880),
                                                  prelu_size=(1,), transpose_b=True, matmul_size=(1, 12),
                                                  num_class=12)
    standalone_dataset = FakeData(size=128, batch_size=32,
                                  image_size=(3, 224, 224), num_classes=12)
    fact = ParallelStrategySearchFactory(standalone_mode_net=standalone_mode_net,
                                         parallel_mode_net=parallel_mode_net)
    fact.mindspore_standalone_impl(dataset=standalone_dataset, epoch=2)
    parallel_dataset = FakeData(size=128, batch_size=4,
                                image_size=(3, 224, 224), use_parallel=True,
                                num_classes=12)
    fact.mindspore_auto_parallel_impl(dataset=parallel_dataset,
                                      epoch=2, device_num=8, auto_parallel_search_mode="recursive_programming")
    fact.checkpoint_cmp(inputs_np=inputs_np)

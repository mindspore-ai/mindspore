import numpy as np

import mindspore as ms
from mindspore import context, Tensor, Parameter, Symbol
from mindspore.nn import Cell, Momentum
from mindspore.ops import operations as P
from mindspore.train import Model
from mindspore.communication.management import get_rank
from mindspore.communication.management import get_group_size
import mindspore.dataset as ds


class FakeDataInitMode:
    RandomInit = 0
    OnesInit = 1
    UniqueInit = 2
    ZerosInit = 3


class GeneratorFakeData:
    """A fake dataset that returns randomly generated images and returns them as PIL images, can be
       applied to dataset_sink_mode=True.
       image data type is np.float32 in default
       label data type is np.float32 in default
       label data is onehot in default

    Args:
        size (int, optional): size of the dataset. Default: 1024 images
        batch_size (int, optional): how many samples per batch to load. Default: 32 images
        image_size(tuple, optional): size if the returned images. Default: (3, 224, 224)
        num_classes(int, optional): number of classes in the dataset. Default: 10
        random_offset (int): offsets the index-based random seed used to
            generate each image. Default: 0

    Examples:
        fake_dataset = GeneratorFakeData()
        ds_train = ds.GeneratorDataset(fake_dataset, ["data", "label"])
        ...
        model = Model(net, loss, opt)
        model.train(epoch_num, ds_train)

    """

    def __init__(self, size=1024, batch_size=32, image_size=(3, 224, 224),
                 num_classes=10, random_offset=0, use_parallel=False,
                 fakedata_mode=FakeDataInitMode.OnesInit, dtype=np.float32):
        self.size = size
        self.rank_batch_size = batch_size
        self.total_batch_size = self.rank_batch_size
        self.random_offset = random_offset
        self.image_size = image_size
        self.num_classes = num_classes
        self.rank_size = 1
        self.rank_id = 0
        self.batch_index = 0
        self.image_data_type = dtype
        self.label_data_type = dtype
        self.is_onehot = True
        self.fakedata_mode = fakedata_mode

        if use_parallel is True:
            self.rank_size = get_group_size()
            self.rank_id = get_rank()

        self.total_batch_size = self.rank_batch_size * self.rank_size
        assert self.size % self.total_batch_size == 0

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

    def create_tuple_iterator(self, num_epochs=-1, do_copy=False):
        return self

    def __next__(self):
        batch_index = self.batch_index
        self.batch_index += 1
        if batch_index * self.total_batch_size >= self.size:
            raise StopIteration

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
        return img_ret, target_ret

    def __len__(self):
        return self.size // self.total_batch_size

    def __iter__(self):
        self.batch_index = 0
        return self

    def reset(self):
        self.batch_index = 0



class Net(Cell):
    def __init__(self, mul_weight, strategy1=None, strategy2=None):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        self.neg = P.Neg().shard(strategy2)
        self.mul_weight = Parameter(mul_weight, "w1")

    def construct(self, x):
        out = self.mul(x, self.mul_weight)
        out = self.neg(out)
        return out


class Lossfn(Cell):
    def __init__(self,):
        super().__init__()
        self.add = P.Add()
        self.reduce = P.ReduceSum()

    def construct(self, x, b):
        out = self.add(x, b)
        out = self.reduce(out)
        return out


_w1 = Tensor(np.ones([1]), dtype=ms.float32)


def compile_net(net, symbol_mode=0):
    learning_rate = 0.1
    momentum = 0.9
    epoch_size = 1
    dataset = ds.GeneratorDataset(
        GeneratorFakeData(size=64, batch_size=8, image_size=(16,), use_parallel=True, num_classes=16),
        ["data", "label"])  # data: [batch_size, image_size], label: [batch_size, num_classes]
    loss = Lossfn()
    opt = Momentum(net.trainable_params(), learning_rate, momentum)

    if symbol_mode == 0:
        s1 = Symbol(divisor=1)
        input_x = Tensor(shape=[s1, 16], dtype=ms.float32)
        label = Tensor(shape=[s1, 16], dtype=ms.float32)

        net.set_inputs(input_x)
        loss.set_inputs(None, label)
    elif symbol_mode == 1:
        s1 = Symbol(divisor=1)
        input_x = Tensor(shape=[None, 16], dtype=ms.float32)
        label = Tensor(shape=[s1, 16], dtype=ms.float32)

        net.set_inputs(input_x)
        loss.set_inputs(None, label)
    elif symbol_mode == 2:
        s1 = Symbol(divisor=1)
        input_x = Tensor(shape=[s1, s1], dtype=ms.float32)
        label = Tensor(shape=[8, s1], dtype=ms.float32)

        net.set_inputs(input_x)
        loss.set_inputs(None, label)

    model = Model(net, loss, optimizer=opt)

    model.train(epoch_size, dataset, dataset_sink_mode=True)
    context.reset_auto_parallel_context()


def test_neg_data_parallel_data_sink():
    '''
    Feature: data sink
    Description: dynamic shape
    Expectation: compile success
    '''
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((8, 1), (1,))
    strategy2 = ((8, 1),)
    net = Net(_w1, strategy1, strategy2)
    compile_net(net)


def test_neg_data_parallel_data_sink_set_dataset_strategy():
    '''
    Feature: data sink
    Description: dynamic shape
    Expectation: compile success
    '''
    s = ((8, 1), (8, 1))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0,
                                      dataset_strategy=s)
    strategy1 = ((8, 1), (1,))
    strategy2 = ((8, 1),)
    net = Net(_w1, strategy1, strategy2)
    compile_net(net)


def test_neg_data_parallel_data_sink_set_dataset_strategy_static_shape():
    '''
    Feature: data sink
    Description: static shape
    Expectation: compile success
    '''
    s = ((8, 1), (8, 1))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0,
                                      dataset_strategy=s)
    strategy1 = ((8, 1), (1,))
    strategy2 = ((8, 1),)
    net = Net(_w1, strategy1, strategy2)
    compile_net(net, symbol_mode=3)


def test_neg_data_parallel_data_sink_set_dataset_strategy_symbol_and_none():
    '''
    Feature: data sink
    Description: use symbol and none to set
    Expectation: compile success
    '''
    s = ((8, 1), (8, 1))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0,
                                      dataset_strategy=s)
    strategy1 = ((8, 1), (1,))
    strategy2 = ((8, 1),)
    net = Net(_w1, strategy1, strategy2)
    compile_net(net, symbol_mode=1)


def test_check_inputs_for_symbol():
    '''
    Feature: data sink
    Description: dynamic shape
    Expectation: compile success
    '''
    s = ((8, 1), (8, 1))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0,
                                      dataset_strategy=s)
    strategy1 = ((8, 1), (1,))
    strategy2 = ((8, 1),)
    net = Net(_w1, strategy1, strategy2)
    compile_net(net, symbol_mode=2)

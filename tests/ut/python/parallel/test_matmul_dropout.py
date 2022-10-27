# Copyright 2019 Huawei Technologies Co., Ltd
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

import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
import mindspore.common.dtype as mstype
from mindspore.common.seed import _get_graph_seed
from mindspore.common.api import _cell_graph_executor
from mindspore._checkparam import Validator
from mindspore.ops.primitive import constexpr
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from tests.ut.python.ops.test_math_ops import VirtualLoss


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


grad_all = C.GradOperation(get_all=True)


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y, b):
        predict = self.network(x, y, b)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y, b):
        return grad_all(self.network)(x, y, b)


@constexpr
def _is_float_dtype(dtype):
    if dtype in [mstype.float32, mstype.float16]:
        return True
    return False

class Dropout(nn.Cell):
    def __init__(self, keep_prob=0.5, dtype=mstype.float32):
        super(Dropout, self).__init__()
        if keep_prob <= 0 or keep_prob > 1:
            raise ValueError("dropout probability should be a number in range (0, 1], but got {}".format(keep_prob))
        Validator.check_subclass("dtype", dtype, mstype.number_type, self.cls_name)
        Validator.check_value_type('keep_prob', keep_prob, [float], self.cls_name)
        self.keep_prob = keep_prob
        seed0, seed1 = _get_graph_seed(0, "dropout")
        self.seed0 = seed0
        self.seed1 = seed1
        self.dtype = dtype
        self.get_shape = P.Shape()
        self.dropout_gen_mask = P.DropoutGenMask(Seed0=self.seed0, Seed1=self.seed1)
        self.dropout_do_mask = P.DropoutDoMask()
        self.cast = P.Cast()
        self.is_gpu = context.get_context('device_target') in ["GPU"]
        self.dropout = P.Dropout(keep_prob)

    def construct(self, x):
        if not self.training:
            return x

        if self.is_gpu:
            out, _ = self.dropout(x)
            return out

        if self.keep_prob == 1:
            return x

        shape = self.get_shape(x)
        dtype = P.DType()(x)
        if _is_float_dtype(dtype):
            keep_prob = self.cast(self.keep_prob, dtype)
        else:
            keep_prob = self.cast(self.keep_prob, mstype.float16)
        output = self.dropout_gen_mask(shape, keep_prob)
        return self.dropout_do_mask(x, output, keep_prob)

    def extend_repr(self):
        return 'keep_prob={}, dtype={}'.format(self.keep_prob, self.dtype)

# model_parallel test
def test_two_matmul_dropout():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, strategy3):
            super().__init__()
            self.matmul1 = P.MatMul().shard(strategy1)
            self.dropout = Dropout()
            self.dropout.dropout_do_mask.shard(strategy2)
            self.dropout.dropout_gen_mask.shard(strategy2)
            self.matmul2 = P.MatMul().shard(strategy3)

        def construct(self, x, y, b):
            out = self.matmul1(x, y)
            out = self.dropout(out)
            out = self.matmul2(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((4, 2), (2, 1))
    strategy2 = ((8, 1),)
    strategy3 = ((1, 8), (8, 1))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2, strategy3)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    net.set_train()
    _cell_graph_executor.compile(net, x, y, b)

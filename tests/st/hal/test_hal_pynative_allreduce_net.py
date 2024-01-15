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
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.ops import operations as P
import mindspore.communication.management as D
from mindspore import context, ops
from mindspore.context import ParallelMode
import mindspore as ms
from mindspore.common.api import _pynative_executor
import time


np.random.seed(1)
os.environ['GLOG_v'] = str(2)

class AllReduceNet(nn.Cell):
    def __init__(self):
        super(AllReduceNet, self).__init__()
        self.mul = P.Mul()
        self.all_reduce = P.AllReduce()
        self.add = P.Add()
        self.s1 = ms.hal.Stream()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, pad_mode='pad')
        self.relu1 = ops.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, pad_mode='pad')
        self.relu2 = ops.ReLU()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, pad_mode='pad')
        self.relu3 = ops.ReLU()
        self.relu4 = ops.ReLU()
        self.conv5 = nn.Conv2d(256, 1024, kernel_size=3, stride=1, padding=1, pad_mode='pad')
        self.relu5 = ops.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def construct(self, x1, y1, x, enable_multi_stream):
        out = self.add(x1, y1)
        if enable_multi_stream:
            with ms.hal.StreamCtx(self.s1):
                out = self.all_reduce(out)
        else:
            out = self.all_reduce(out)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool(x)
        x = x*2
        if enable_multi_stream:
            self.s1.synchronize()
        out = self.mul(out, x)
        return out

def msrun_train_allreduce_8p(enable_multi_stream):
    context.set_context(mode=context.PYNATIVE_MODE)
    D.init()
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=False,
                                      device_num=D.get_group_size())

    net = AllReduceNet()
    x1 = Tensor(np.ones([16, 1024, 64, 64]).astype(np.float32), mstype.float32)
    y1 = Tensor(np.array(np.random.randn(16, 1024, 64, 64))).astype(np.float32)
    x = Tensor(np.random.rand(16, 3, 128, 128), ms.float32)
    net(x1, y1, x, enable_multi_stream)
    net(x1, y1, x, enable_multi_stream)
    _pynative_executor.sync()

    t1 = time.time()
    for _ in range(1000):
        net(x1, y1, x, enable_multi_stream)
    _pynative_executor.sync()
    return time.time() - t1

multi_stream_time = msrun_train_allreduce_8p(True)
single_stream_time = msrun_train_allreduce_8p(False)
print("multi_stream_time:", multi_stream_time, flush=True)
print("single_stream_time:", single_stream_time, flush=True)

assert single_stream_time > multi_stream_time

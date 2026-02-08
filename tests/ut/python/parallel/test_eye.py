# Copyright 2025 Huawei Technologies Co., Ltd
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

import mindspore.nn as nn
from mindspore import dtype as mstype
from mindspore import context, Layout
from mindspore.graph.api import _cell_graph_executor
from mindspore.ops import operations as P


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(nn.Cell):
    def __init__(self, in_strategy=None, out_strategy=None, self_define_shard=False):
        super().__init__()
        self.eye = P.Eye().shard(in_strategy, out_strategy)
        if self_define_shard:
            self.eye.add_prim_attr("self_define_shard", True)
        self.dtype = mstype.float32

    def construct(self, n, m):
        out = self.eye(n, m, self.dtype)
        return out


def compile_net(net, *inputs):
    net.set_train()
    _cell_graph_executor.compile(net, *inputs)


def common_train_compile(*inputs, **kwargs):
    net = Net(**kwargs)
    compile_net(net, *inputs)
    context.reset_auto_parallel_context()


def test_eye_standalone():
    """
    Feature: distribute operator eye with standalone info.
    Description: eye net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    n, m = 4, 4
    common_train_compile(n, m)


def test_eye_self_define_shard():
    """
    Feature: distribute operator eye with self_define shard.
    Description: eye net with self_define strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    dev_num = 64
    context.set_auto_parallel_context(device_num=dev_num, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    n, m = 4, 4
    layout = Layout((dev_num,), ("dev",))
    common_train_compile(n, m, in_strategy=(), out_strategy=(layout("None", "None"),), self_define_shard=True)

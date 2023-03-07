# Copyright 2023 Huawei Technologies Co., Ltd
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
import mindspore.context as context
import mindspore.ops.operations as ops
from mindspore.ops.operations import _inner_ops as inner

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class FnDict:
    def __init__(self):
        self.fn_dict = {}

    def __call__(self, fn):
        self.fn_dict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fn_dict.get(name)


def test_dyn_broadcast(tag):
    """
    Feature: ConvertDynmicBroadcastTo Pass
    Description: ConvertDynmicBroadcastTo rewrite graph.
    Expectation: Get correct Graph.
    """
    fns = FnDict()
    d_shape = ops.TensorShape()
    d_broadcastto = inner.DynamicBroadcastTo()

    @fns
    def before(data, shape):
        shape = d_shape(shape)
        return d_broadcastto(data, shape)

    return fns[tag]

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
""" test sparse feature bprop """
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore.common import dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.ops import composite as C
from mindspore.ops.operations.comm_ops import AllReduce, _MirrorOperator
from mindspore.ops._grad.grad_base import bprop_getters
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel
from mindspore.ops.primitive import prim_attr_register, PrimitiveWithInfer
from mindspore.common.api import _executor
from mindspore.communication.management import HCCL_WORLD_COMM_GROUP

class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x):
        return C.grad_all(self.network)(x)

class VirtualGatherV2(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self):
        """init index_select"""
        super(VirtualGatherV2, self).__init__('VirtualGatherV2')
        self.init_prim_io_names(inputs=['params', 'indices', 'axis'], outputs=['output'])

    def __infer__(self, params, indices, axis):
        validator.check_subclass("params", params['dtype'], mstype.tensor, self.name)
        validator.check_tensor_type_same({"indices": indices['dtype']}, mstype.int_type, self.name)
        validator.check_subclass("axis", axis['dtype'], mstype.int_, self.name)
        axis_v = axis['value']
        params_shp = params['shape']
        rank = len(params_shp)
        validator.check_int_range("axis", axis_v, -rank, rank, Rel.INC_LEFT, self.name)
        if axis_v < 0:
            axis_v += rank
        out_shape = params_shp[:axis_v] + indices['shape'] + params_shp[axis_v + 1:]
        out = {'shape': out_shape,
               'dtype': params['dtype'],
               'value': None}
        return out

@bprop_getters.register(VirtualGatherV2)
def get_bprop_gather_v2(self):
    """Generate bprop for GatherV2"""

    def bprop(x, indices, axis, out, dout):
        return (indices, dout, x), axis, out

    return bprop

def test_bprop_with_sparse_feature_allreduce():
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="hybrid_parallel")

    class Net(nn.Cell):
        def __init__(self, axis=0, shape=None):
            super(Net, self).__init__()
            if shape is None:
                shape = [8, 8]
            self.all_reduce = AllReduce()
            self.gatherv2 = VirtualGatherV2()
            self.index = Tensor(np.ones(shape), dtype=ms.int32)
            self.axis = axis

        def construct(self, x):
            out = self.all_reduce(x)
            out = self.gatherv2(out, self.index, self.axis)

            return out

    net = GradWrap(Net())
    x = Tensor(np.ones([64, 64]), dtype=ms.float32)

    _executor.compile(net, x)

def test_bprop_with_sparse_feature_mirror():
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="hybrid_parallel")

    class Net(nn.Cell):
        def __init__(self, axis=0, shape=None):
            super(Net, self).__init__()
            if shape is None:
                shape = [8, 8]
            self.mirror = _MirrorOperator(group=HCCL_WORLD_COMM_GROUP)
            self.gatherv2 = VirtualGatherV2()
            self.index = Tensor(np.ones(shape), dtype=ms.int32)
            self.axis = axis

        def construct(self, x):
            out = self.mirror(x)
            out = self.gatherv2(out, self.index, self.axis)

            return out

    net = GradWrap(Net())
    x = Tensor(np.ones([64, 64]), dtype=ms.float32)

    _executor.compile(net, x)

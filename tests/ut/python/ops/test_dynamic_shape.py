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
""" test dynamic shape """
import numpy as np

from mindspore import Tensor, context, nn, Parameter
from mindspore import dtype as mstype
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, save_graphs=False)


def test_sparse_apply_proximal_ada_grad():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.sparse_apply_proximal_adagrad = P.SparseApplyProximalAdagrad()
            self.var = Parameter(Tensor(np.random.rand(7800, 80).astype(np.float32)), name="var")
            self.accum = Parameter(Tensor(np.random.rand(7800, 80).astype(np.float32)), name="accum")
            self.lr = 0.01
            self.l1 = 0.0
            self.l2 = 0.0

        def construct(self, grad, indices):
            out = self.sparse_apply_proximal_adagrad(self.var, self.accum, self.lr, self.l1, self.l2, grad, indices)
            return out[0]

    class NetWrapper(nn.Cell):
        def __init__(self):
            super(NetWrapper, self).__init__()
            self.unq = P.Unique()
            self.add = P.Add()
            self.expand_dims = P.ExpandDims()
            self.cast = P.Cast()
            self.net = Net()

        def construct(self, grad, inp):
            ids, _ = self.unq(inp)
            new_grad = self.expand_dims(ids, 1)
            new_grad = self.cast(new_grad, mstype.float32) + grad
            return self.net(new_grad, ids)

    net = NetWrapper()
    grad = Tensor(np.random.rand(1, 80).astype(np.float32))
    indices = Tensor(np.ones([7800]), mstype.int32)
    net(grad, indices)


def test_sparse_apply_ftrl():
    class SparseApplyFtrlNet(nn.Cell):
        def __init__(self):
            super(SparseApplyFtrlNet, self).__init__()
            self.sparse_apply_ftrl = P.SparseApplyFtrl(lr=0.01, l1=0.0, l2=0.0, lr_power=-0.5)
            self.var = Parameter(Tensor(np.random.rand(7800, 80).astype(np.float32)), name="var")
            self.accum = Parameter(Tensor(np.random.rand(7800, 80).astype(np.float32)), name="accum")
            self.linear = Parameter(Tensor(np.random.rand(7800, 80).astype(np.float32)), name="linear")

        def construct(self, grad, indices):
            out = self.sparse_apply_ftrl(self.var, self.accum, self.linear, grad, indices)
            return out[0]

    class NetWrapper(nn.Cell):
        def __init__(self):
            super(NetWrapper, self).__init__()
            self.unq = P.Unique()
            self.add = P.Add()
            self.expand_dims = P.ExpandDims()
            self.cast = P.Cast()
            self.net = SparseApplyFtrlNet()

        def construct(self, grad, inp):
            ids, _ = self.unq(inp)
            new_grad = self.expand_dims(ids, 1)
            new_grad = self.cast(new_grad, mstype.float32) + grad
            return self.net(new_grad, ids)

    net = NetWrapper()
    grad = Tensor(np.random.rand(1, 80).astype(np.float32))
    indices = Tensor(np.ones([7800]), mstype.int32)
    net(grad, indices)


def test_gatherv2():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.unq = P.Unique()
            self.gather = P.Gather()
            self.yy = Tensor(np.ones([8], dtype=np.int32))

        def construct(self, x, y):
            shp = P.Shape()(self.yy)
            y = P.Reshape()(y, shp)
            u, _ = self.unq(y)
            u_shp = P.DynamicShape()(u)
            z = self.gather(x, u, 0)
            return z, u_shp

    x = Tensor(np.ones([20, 12], dtype=np.float32))
    y = Tensor(np.ones([2, 4], dtype=np.int32))
    net = Net()
    net(x, y)


def test_segmentsum():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.unq = P.Unique()
            self.segment_ids = Tensor([0, 0, 1, 2, 1, 1, 1, 1], mstype.int32)
            self.sum = P.UnsortedSegmentSum()
        def construct(self, x):
            u, _ = self.unq(x)
            shp = P.DynamicShape()(u)
            z = self.sum(x, self.segment_ids, shp[0])
            return z, shp[0]

    x = Tensor(np.ones([8], dtype=np.int32))
    net = Net()
    net(x)


def test_addn():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.unq = P.Unique()
            self.addn = P.AddN()

        def construct(self, x):
            u, _ = self.unq(x)
            u = self.addn((u, u, u))
            z = self.addn([u, u])
            return z

    y = Tensor(np.ones([8], dtype=np.int32))
    net = Net()
    net(y)

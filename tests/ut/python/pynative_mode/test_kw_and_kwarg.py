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
""" test dtype and shape as attr"""
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore import dtype as mstype
from mindspore.ops.composite import base as C


def test_kw_nested():
    class NetKeyValueArg(nn.Cell):
        def __init__(self):
            super().__init__()

        def construct(self, x, y, *arg, w, **kwargs):
            return x + y + arg[0] + w + kwargs['c']

    class NetOut(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.in_net = net

        def construct(self, x, y, z):
            ret = self.in_net(x, y, z, w=x, a=x, b=y, c=z) + x
            return ret

    in_net = NetKeyValueArg()
    out_net = NetOut(in_net)
    x = Tensor(np.ones([3, 4, 5], np.float32))
    y = Tensor(np.zeros([3, 4, 5], np.int32))
    z = Tensor(np.ones([3, 4, 5], np.float64))
    context.set_context(mode=context.PYNATIVE_MODE)

    ret = out_net(x, y, z)
    assert ret.dtype == mstype.float64
    assert ret.shape == (3, 4, 5)
    assert (ret.asnumpy() == np.ones([3, 4, 5], np.float64) * 5).all()


def test_kw_grad():
    class KwNet(nn.Cell):
        def __init__(self):
            super(KwNet, self).__init__()

        def construct(self, x, y, *arg, **kwargs):
            return 2 * x + 3 * y + 4 * arg[0] + 5 * kwargs['v']

    class GradKwNet(nn.Cell):
        def __init__(self, net):
            super(GradKwNet, self).__init__()
            self.net = net
            self.grad_all_wit_sense = C.GradOperation(get_all=True, sens_param=True)

        def construct(self, x, y, *arg, **kwargs):
            return self.grad_all_wit_sense(self.net)(x, y, *arg, **kwargs)

    kw_net = KwNet()
    x = Tensor(np.ones([1, 2, 3], np.int32))
    y = Tensor(np.ones([1, 2, 3], np.float32))
    z = Tensor(np.ones([1, 2, 3], np.float64))
    u = Tensor(np.ones([1, 2, 3], np.float16))
    v = Tensor(np.ones([1, 2, 3], np.int32))
    w = Tensor(np.ones([1, 2, 3], np.float64))
    sens = Tensor(np.ones([1, 2, 3], np.float64))
    context.set_context(mode=context.PYNATIVE_MODE)

    kw_net.set_grad(True)
    ret = kw_net(x, y, z, u=u, v=v, w=w)
    assert (ret.asnumpy() == np.ones([1, 2, 3], np.float64) * 14).all()

    grad_kw_net = GradKwNet(kw_net)
    ret_grad = grad_kw_net(x, y, z, u=u, v=v, w=w, sens=sens)
    assert len(ret_grad) == 6
    assert (ret_grad[0].asnumpy() == np.ones([1, 2, 3]) * 2).all()
    assert ret_grad[0].dtype == mstype.int32
    assert (ret_grad[1].asnumpy() == np.ones([1, 2, 3]) * 3).all()
    assert ret_grad[1].dtype == mstype.float32
    assert (ret_grad[2].asnumpy() == np.ones([1, 2, 3]) * 4).all()
    assert ret_grad[2].dtype == mstype.float64
    assert (ret_grad[3].asnumpy() == np.zeros([1, 2, 3])).all()
    assert ret_grad[3].dtype == mstype.float16
    assert (ret_grad[4].asnumpy() == np.ones([1, 2, 3]) * 5).all()
    assert ret_grad[4].dtype == mstype.int32
    assert (ret_grad[5].asnumpy() == np.zeros([1, 2, 3])).all()
    assert ret_grad[5].dtype == mstype.float64


def test_grad():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self, x, y, z):
            return 2 * x + 3 * y + 4 * z

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.grad_all_wit_sense = C.GradOperation(get_all=True, sens_param=True)

        def construct(self, x, y, z, sens):
            return self.grad_all_wit_sense(self.net)(x, y, z, sens)

    net = Net()
    x = Tensor(np.ones([1, 2, 3], np.int32))
    y = Tensor(np.ones([1, 2, 3], np.float32))
    z = Tensor(np.ones([1, 2, 3], np.float16))
    sens = Tensor(np.ones([1, 2, 3], np.float32))
    context.set_context(mode=context.PYNATIVE_MODE)

    net.set_grad(True)
    ret = net(x, y, z)
    assert (ret.asnumpy() == np.ones([1, 2, 3], np.float64) * 9).all()

    grad_net = GradNet(net)
    ret_grad = grad_net(x, y, z, sens)
    assert len(ret_grad) == 3
    assert (ret_grad[0].asnumpy() == np.ones([1, 2, 3]) * 2).all()
    assert ret_grad[0].dtype == mstype.int32
    assert (ret_grad[1].asnumpy() == np.ones([1, 2, 3]) * 3).all()
    assert ret_grad[1].dtype == mstype.float32
    assert (ret_grad[2].asnumpy() == np.ones([1, 2, 3]) * 4).all()
    assert ret_grad[2].dtype == mstype.float16

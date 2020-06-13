import logging
import numpy as np
import mindspore.context as context
import mindspore.ops.composite as C
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.nn.composite_ops import LambUpdateWithLR, LambNextMV
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype

log = logging.getLogger("ME")
log.setLevel(level=logging.DEBUG)
context.set_context(mode=context.GRAPH_MODE, save_graphs=True, device_target="Ascend")

class LambUpdateNet(Cell):
    def __init__(self,shape):
        super(LambUpdateNet, self).__init__()
        self.lamb_update = LambUpdateWithLR()
        self.x6 = Parameter(initializer('normal', shape), name='x6')

    def construct(self, x1, x2, x3, x4, x5, gy, se, my):
        return self.lamb_update(x1, x2, x3, x4, x5, self.x6, gy, se, my)

class LambUpdateNetTbe(Cell):
    def __init__(self):
        super(LambUpdateNetTbe, self).__init__()
        self.mul = P.Mul()
        self.sqrt = P.Sqrt()
        self.rsqrt = P.Rsqrt()
        self.square = P.Square()
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.pow = P.Pow()
        self.select = P.Select()
        self.greater = P.Greater()
        self.fill = P.Fill()
        self.dtype = P.DType()

    def construct(self, x1, x2, x3, x4, x5, x6, gy, se, my):
        trust_ratio = self.select(
            self.greater(x2, gy),
            self.select(self.greater(x1, gy), x2 / x3, se),
            se
        )
        trust_ratio = C.clip_by_value(trust_ratio, gy, my)
        update_with_lr = self.mul(self.mul(trust_ratio, x4), x5)
        next_param = x6 - self.reshape(update_with_lr, self.shape(x6))
        return next_param

class LambNextMVNet(Cell):
    def __init__(self, shape):
        super(LambNextMVNet, self).__init__()
        self.i2 = Parameter(initializer('normal', shape), name='i2')
        self.i5 = Parameter(initializer('normal', shape), name='i5')
        self.lamb_next = LambNextMV()

    def construct(self, i1, i3, i4, i6, i7, i8, i9, x0, x1, x2, x3):
        return self.lamb_next(i1, self.i2, i3, i4, self.i5, i6, i7, i8, i9, x0, x1, x2, x3)

class LambNextMVNetTbe(Cell):
    def __init__(self):
        super(LambNextMVNetTbe, self).__init__()
        self.mul = P.Mul()
        self.sqrt = P.Sqrt()
        self.rsqrt = P.Rsqrt()
        self.square = P.Square()
        self.cast = P.Cast()
        self.reshape = P.Reshape
        self.pow = P.Pow()
        self.select = P.Select()

    def construct(self, i1, i2, i3, i4, i5, i6, i7, i8, i9, x0, x1, x2, x3):
        # x1: 1 - beta2     i1: g^2         x0: beta2
        # i2: v             i9: 1 - beta1   i4: g
        # i8: beta1         i5: m           i6: 1 - beta1^(gs + 1)
        # i3: 1 - beta2^(gs + 1)            x3: eps
        # x2: weight_decay_tensor           i7: param
        m_fp32 = self.cast(i5, mstype.float32)
        v_fp32 = self.cast(i2, mstype.float32)
        next_m = self.mul(i8, m_fp32) + self.mul(i9, i4)
        next_v = self.mul(x0, v_fp32) + self.mul(x1, i1)
        next_mm = next_m / i6
        next_vv = next_v / i3
        update = next_mm / (self.sqrt(next_vv) + x3)
        add3 = self.mul(next_mm, self.rsqrt(next_vv + x3)) + x2 * i7
        return add3, next_m, next_v, update


def tensor_all(*args):
    res = [Tensor(a) for a in args]
    return res

# composite not inline funcGraph
# def test_composite_lamb_update_with_lr():
#     shape = [1, 16]
#     oshape = [1]
#     x1 = np.random.normal(0, 1, oshape).astype(np.float32)
#     x2 = np.random.normal(0, 1, oshape).astype(np.float32)
#     x3 = np.random.normal(0, 1, oshape).astype(np.float32)
#     x4 = np.random.normal(0, 1, oshape).astype(np.float32)
#     x5 = np.random.normal(0, 1, shape).astype(np.float32)
#     gy = np.random.normal(0, 1, oshape).astype(np.float32)
#     se = np.random.normal(0, 1, oshape).astype(np.float32)
#     my = np.random.normal(0, 1, oshape).astype(np.float32)

#     net = LambUpdateNet(shape)
#     net1 = LambNextMVNetTbe()

#     x6 = net.x6.data.asnumpy().copy()

#     tx1, tx2, tx3, tx4, tx5, tx6, tgy, tse, tmy = tensor_all(x1, x2, x3, x4, x5, x6, gy, se, my)

#     _ = net(tx1, tx2, tx3, tx4, tx5, tgy, tse, tmy)
#     tres = net1(tx1, tx2, tx3, tx4, tx5, tx6, tgy, tse, tmy)

#     ares = net.x6.data.asnumpy().copy()

#     print("=======================================")
#     print("x6 before:\n{}".formata(x6))
#     print("white res:\n{}".format(ares)) # x6 will be inplace change
#     print("tbe b res:\n{}".format(tres))
#     print("=======================================")


# def test_composite_lamb_next_mv():
#     shape = [1, 16]
#     i1 = np.random.normal(0, 1, shape).astype(np.float32)
#     i3 = np.random.normal(0, 1, shape).astype(np.float32)
#     i4 = np.random.normal(0, 1, shape).astype(np.float32)
#     i6 = np.random.normal(0, 1, shape).astype(np.float32)
#     i7 = np.random.normal(0, 1, shape).astype(np.float32)
#     i8 = np.random.normal(0, 1, shape).astype(np.float32)
#     i9 = np.random.normal(0, 1, shape).astype(np.float32)
#     x0 = np.random.normal(0, 1, shape).astype(np.float32)
#     x1 = np.random.normal(0, 1, shape).astype(np.float32)
#     x2 = np.random.normal(0, 1, shape).astype(np.float32)
#     x3 = np.random.normal(0, 1, shape).astype(np.float32)

#     net = LambNextMVNet(shape)
#     net1 = LambNextMVNetTbe()

#     i2 = net.i2.data.asnumpy().copy()
#     i5 = net.i5.data.asnumpy().copy()

#     ti1, ti2, ti3, ti4, ti5, ti6, ti7, ti8, ti9, tx0, tx1, tx2, tx3 = \
#         tensor_all(i1, i2, i3, i4, i5, i6, i7, i8, i9, x0, x1, x2, x3)

#     wa3, wup = net(ti1, ti3, ti4, ti6, ti7, ti8, ti9, tx0, tx1, tx2, tx3)
#     ba3, ba0, ba1, bup = net1(ti1, ti2, ti3, ti4, ti5, ti6, ti7, ti8, ti9, tx0, tx1, tx2, tx3)
#     wi2 = net.i2.data.asnumpy().copy()
#     wi5 = net.i5.data.asnumpy().copy()

#     print("==========================================")
#     print("before: \ni2:\n{}\ni5:\n{}".format(i2, i5))
#     print("wa3:{}\nwi2:\n{}\nwi5:\n{}\nwup:\n{}".format(wa3, wi2, wi5, wup))
#     print("ba3:{}\nbi2:\n{}\nbi5:\n{}\nbup:\n{}".format(ba3, ba0, ba1, bup))

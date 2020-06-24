import pytest
import numpy as np
import mindspore.context as context
from mindspore import Tensor, Parameter
from mindspore.nn import Cell
from mindspore.nn.graph_kernels import LambUpdateWithLR, LambNextMV

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

class LambNet(Cell):
    def __init__(self, i2, i5, x6):
        super(LambNet, self).__init__()
        self.i2 = Parameter(i2, name='i2')
        self.i5 = Parameter(i5, name='i5')
        self.x6 = Parameter(x6, name='x6')
        self.lamb_next = LambNextMV()
        self.lamb_update = LambUpdateWithLR()

    def construct(self, i1, i3, i4, i6, i7, i8, i9, ix0, ix1, ix2, ix3,
                  x1, x2, x3, x4, x5, gy, se, my):
        return self.lamb_next(i1, self.i2, i3, i4, self.i5, i6, i7, i8, i9, ix0,
                              ix1, ix2, ix3), \
               self.lamb_update(x1, x2, x3, x4, x5, self.x6, gy, se, my)

def LambUpdateNumpy(x1, x2, x3, x4, x5, x6, gy, se, my):
    trust_ratio = np.where(np.greater(x2, gy),
                           np.where(np.greater(x1, gy), np.divide(x2, x3), se),
                           se)
    trust_ratio = np.maximum(np.minimum(trust_ratio, my), gy)
    update_with_lr = trust_ratio * x4 * x5
    next_param = x6 - np.reshape(update_with_lr, x6.shape)
    return next_param

def LambNextMVNumpy(i1, i2, i3, i4, i5, i6, i7, i8, i9, x0, x1, x2, x3):
    m_fp32 = i5.astype(np.float32)
    v_fp32 = i2.astype(np.float32)
    next_m = i8 * m_fp32 + i9 * i4
    next_v = x0 * v_fp32 + x1 * i1
    next_mm = next_m / i6
    next_vv = next_v / i3
    update = next_mm / (np.sqrt(next_vv) + x3)
    add3 = next_mm / np.sqrt(next_vv + x3) + x2 * i7
    return add3, next_m, next_v, update


def tensor_all(*args):
    res = [Tensor(a) for a in args]
    return res

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_graph_kernel_lamb():
    shape = [1, 16]
    oshape = [1]
    np.random.seed(0)
    x1 = np.random.normal(0, 1, oshape).astype(np.float32)
    x2 = np.random.normal(0, 1, oshape).astype(np.float32)
    x3 = np.random.normal(0, 1, oshape).astype(np.float32)
    x4 = np.random.normal(0, 1, oshape).astype(np.float32)
    x5 = np.random.normal(0, 1, shape).astype(np.float32)
    x6 = np.random.normal(0, 1, shape).astype(np.float32)
    gy = np.random.normal(0, 1, oshape).astype(np.float32)
    se = np.random.normal(0, 1, oshape).astype(np.float32)
    my = np.random.normal(0, 1, oshape).astype(np.float32)

    tx1, tx2, tx3, tx4, tx5, tx6, tgy, tse, tmy = tensor_all(
        x1, x2, x3, x4, x5, x6, gy, se, my)

    np.random.seed(1)
    i1 = np.abs(np.random.normal(0, 1, shape)).astype(np.float32)
    i2 = np.abs(np.random.normal(0, 1, shape)).astype(np.float32)
    i3 = np.abs(np.random.normal(0, 1, shape)).astype(np.float32)
    i4 = np.random.normal(0, 1, shape).astype(np.float32)
    i5 = np.random.normal(0, 1, shape).astype(np.float32)
    i6 = np.abs(np.random.normal(0, 1, shape)).astype(np.float32)
    i7 = np.random.normal(0, 1, shape).astype(np.float32)
    i8 = np.random.normal(0, 1, shape).astype(np.float32)
    i9 = np.random.normal(0, 1, shape).astype(np.float32)
    ix0 = np.abs(np.random.normal(0, 1, shape)).astype(np.float32)
    ix1 = np.abs(np.random.normal(0, 1, shape)).astype(np.float32)
    ix2 = np.random.normal(0, 1, shape).astype(np.float32)
    ix3 = np.ones(shape).astype(np.float32) * 1e-6

    ti1, ti2, ti3, ti4, ti5, ti6, ti7, ti8, ti9, tix0, tix1, tix2, tix3 = \
        tensor_all(i1, i2, i3, i4, i5, i6, i7, i8, i9, ix0, ix1, ix2, ix3)

    context.set_context(enable_graph_kernel=True)

    net = LambNet(ti2, ti5, tx6)
    (wa3, wup), _ = net(ti1, ti3, ti4, ti6, ti7, ti8, ti9, tix0, tix1, tix2, tix3,
                        tx1, tx2, tx3, tx4, tx5, tgy, tse, tmy)

    wi2 = net.i2.data.asnumpy().copy()
    wi5 = net.i5.data.asnumpy().copy()
    ares = net.x6.data.asnumpy().copy()

    context.set_context(enable_graph_kernel=False)

    a3, a0, a1, up = LambNextMVNumpy(i1, i2, i3, i4, i5, i6, i7, i8, i9, ix0,
                                     ix1, ix2, ix3)

    np_res = LambUpdateNumpy(x1, x2, x3, x4, x5, x6, gy, se, my)

    rtol = 0.0001
    atol = 0.0001

    wres = (wa3.asnumpy().copy(), wi5, wi2, wup.asnumpy().copy())
    bres = (a3, a0, a1, up)

    cmp_res = list(map(lambda x, y: np.allclose(x, y, rtol, atol),
                       wres, bres))

    assert all(cmp_res) and np.allclose(ares, np_res, rtol, atol)

from tests.mark_utils import arg_mark
import numpy as np
import pytest
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops.operations.nn_ops import SparseApplyMomentum
import mindspore.common.dtype as mstype
import mindspore as ms

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")


class Net(nn.Cell):
    def __init__(self, use_nesterov):
        super(Net, self).__init__()
        self.sparse_apply_momentum = SparseApplyMomentum(use_nesterov=use_nesterov)

    def construct(self, var, accum, lr, grad, indices, momentum):
        out = self.sparse_apply_momentum(var, accum, lr, grad, indices, momentum)
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparseapplymomentum_fp16_int32():
    """
    Feature: SparseApplyAdagrad gpu op
    Description: Test output for fp32 dtype
    Expectation: Output matching expected values
    """
    var = Tensor(np.array([[4.1, 7.2], [1.1, 3.0]]).astype(np.float16))
    accum = Tensor(np.array([[2.2, 3.0], [3.1, 0.5]]).astype(np.float16))
    lr = Tensor(0.01, mstype.float16)
    grad = Tensor(np.array([[0.3, 0.2], [0.4, 0.1]]).astype(np.float16))
    indices = Tensor([0, 1], ms.int32)
    momentum = Tensor(0.99, mstype.float16)
    sparse_apply_momentum = Net(use_nesterov=False)
    var_out = sparse_apply_momentum(var, accum, lr, grad, indices, momentum)
    var_expect = np.array([[4.08, 7.168], [1.064, 2.994]]).astype(np.float16)
    assert np.all(var_out.asnumpy() == var_expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparseapplymomentum_fp32_int32():
    """
    Feature: SparseApplyAdagrad gpu op
    Description: Test output for fp32 dtype
    Expectation: Output matching expected values
    """
    var = Tensor(np.array([[4.1, 7.2], [1.1, 3.0]]).astype(np.float32))
    accum = Tensor(np.array([[2.2, 3.0], [3.1, 0.5]]).astype(np.float32))
    lr = Tensor(0.01, mstype.float32)
    grad = Tensor(np.array([[0.3, 0.2], [0.4, 0.1]]).astype(np.float32))
    indices = Tensor([0, 1], ms.int32)
    momentum = Tensor(0.99, mstype.float32)
    sparse_apply_momentum = Net(use_nesterov=False)
    var_out = sparse_apply_momentum(var, accum, lr, grad, indices, momentum)
    var_expect = np.array([[4.07522, 7.1682997], [1.06531, 2.99405]]).astype(np.float32)
    assert np.all(var_out.asnumpy() == var_expect)

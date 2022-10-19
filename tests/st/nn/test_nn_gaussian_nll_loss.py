import numpy as np
import pytest
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context


class Net(nn.Cell):
    def __init__(self, full=False, reduction='mean'):
        super(Net, self).__init__()
        self.loss = nn.GaussianNLLLoss(full=full, reduction=reduction)

    def construct(self, x, label, v):
        loss = self.loss(x, label, v)
        return loss


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('full', [True, False])
def test_gaussian_nll_loss_full(mode, full):
    """
    Feature: GaussianNLLLoss with reduction='mean'
    Description: Verify the result of GaussianNLLLoss
    Expectation: success
    """
    context.set_context(mode=mode)
    net = Net(full=full)
    arr1 = np.arange(8).reshape((4, 2))
    arr2 = np.array([2, 3, 1, 4, 6, 4, 4, 9]).reshape((4, 2))
    a = Tensor(arr1, mstype.float32)
    b = Tensor(arr2, mstype.float32)
    var = Tensor(np.ones((4, 1)), mstype.float32)
    output = net(a, b, var)
    if full:
        expected = np.array(2.35644, np.float32)
    else:
        expected = np.array(1.4375, np.float32)
    assert np.allclose(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('reduction', ['mean', 'sum', 'none'])
def test_gaussian_nll_loss_reduction(mode, reduction):
    """
    Feature: GaussianNLLLoss with full=False
    Description: Verify the result of GaussianNLLLoss
    Expectation: success
    """
    context.set_context(mode=mode)
    net = Net(reduction=reduction)
    arr1 = np.arange(8).reshape((4, 2))
    arr2 = np.array([2, 3, 1, 4, 6, 4, 4, 9]).reshape((4, 2))
    a = Tensor(arr1, mstype.float32)
    b = Tensor(arr2, mstype.float32)
    var = Tensor(np.ones((4, 1)), mstype.float32)
    output = net(a, b, var)
    if reduction == 'mean':
        expected = np.array(1.4375, np.float32)
    elif reduction == 'sum':
        expected = np.array(11.5, np.float32)
    else:
        expected = np.array([[2.0000, 2.0000], [0.5000, 0.5000],
                             [2.0000, 0.5000], [2.0000, 2.0000]], np.float32)
    assert np.allclose(output.asnumpy(), expected)

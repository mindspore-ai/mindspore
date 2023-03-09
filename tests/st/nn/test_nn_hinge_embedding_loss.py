import numpy as np
import pytest
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context


class Net(nn.Cell):
    def __init__(self, reduction='mean'):
        super(Net, self).__init__()
        self.loss = nn.HingeEmbeddingLoss(margin=1.0, reduction=reduction)

    def construct(self, x, label):
        loss = self.loss(x, label)
        return loss


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('reduction', ['mean', 'sum', 'none'])
def test_hinge_embedding_loss(mode, reduction):
    """
    Feature: HingeEmbeddingLoss with margin=1.0
    Description: Verify the result of HingeEmbeddingLoss
    Expectation: success
    """
    context.set_context(mode=mode)
    net = Net(reduction=reduction)
    arr1 = np.array([0.9, -1.2, 2, 0.8, 3.9, 2, 1, 0, -1]).reshape((3, 3))
    arr2 = np.array([1, 1, -1, 1, -1, 1, -1, 1, 1]).reshape((3, 3))
    a = Tensor(arr1, mstype.float32)
    b = Tensor(arr2, mstype.float32)
    output = net(a, b)

    if reduction == 'mean':
        expected = np.array(1 / 6, np.float32)
    elif reduction == 'sum':
        expected = np.array(1.5, np.float32)
    else:
        expected = np.array([[0.9000, -1.2000, 0.0000],
                             [0.8000, 0.0000, 2.0000],
                             [0.0000, 0.0000, -1.0000]], np.float32)
    assert np.allclose(output.asnumpy(), expected)

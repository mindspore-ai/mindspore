import numpy as np
import pytest
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import context


class Net(nn.Cell):
    def __init__(self, reduction='mean'):
        super(Net, self).__init__()
        self.reduction = reduction

    def construct(self, x, label):
        loss = ops.hinge_embedding_loss(x, label, reduction=self.reduction)
        return loss


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_hinge_embedding_loss_abnormal(mode):
    """
    Feature: HingeEmbeddingLoss
    Description: Verify abnormal inputs of HingeEmbeddingLoss
    Expectation: raise ValueError
    """
    context.set_context(mode=mode)
    net = Net(reduction='mean')
    arr1 = np.array([0.9, -1.2, 2, 0.8, 3.9, 2, 1, 0, -1]).reshape((3, 3))
    arr2 = np.array([1, 1, -1, 1]).reshape((2, 2))
    a = Tensor(arr1, mstype.float32)
    b = Tensor(arr2, mstype.float32)
    with pytest.raises(ValueError):
        net(a, b)

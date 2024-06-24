from tests.mark_utils import arg_mark
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

    def construct(self, input1, input2, target):
        loss = ops.cosine_embedding_loss(input1, input2, target, reduction=self.reduction)
        return loss


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('reduction', ['mean', 'sum', 'none'])
def test_cosine_embedding_loss(mode, reduction):
    """
    Feature: cosine_embedding_loss
    Description: Verify the result of cosine_embedding_loss
    Expectation: success
    """
    context.set_context(mode=mode)
    net = Net(reduction=reduction)
    intput1 = Tensor(np.array([[0.3, 0.8], [0.4, 0.3]]), mstype.float32)
    intput2 = Tensor(np.array([[0.4, 1.2], [-0.4, -0.9]]), mstype.float32)
    target = Tensor(np.array([1, -1]), mstype.float32)
    output = net(intput1, intput2, target)

    if reduction == 'mean':
        expected = np.array(0.0003426075, np.float32)
    elif reduction == 'sum':
        expected = np.array(0.0006852150, np.float32)
    else:
        expected = np.array([0.0006852150, 0.0000], np.float32)
    assert np.allclose(output.asnumpy(), expected, 0.0001, 0.0001)

import numpy as np
import pytest
from mindspore import nn
from mindspore import context
from mindspore.ops.operations.array_ops import SegmentMean
from mindspore import Tensor
from .test_grad_of_dynamic import TestDynamicGrad


class NetSegmentMean(nn.Cell):
    def __init__(self):
        super(NetSegmentMean, self).__init__()
        self.segmentmean = SegmentMean()

    def construct(self, x, segment_ids):
        return self.segmentmean(x, segment_ids)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dynamic_segment_mean_shape():
    """
    Feature: SegmentMean Grad DynamicShape.
    Description: Test case of dynamic shape for SegmentMean grad operator on CPU, GPU and Ascend.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    test_dynamic = TestDynamicGrad(NetSegmentMean())
    x = Tensor(np.array([2, 2, 3, 4]).astype(np.float32))
    segment_ids = Tensor(np.array([0, 0, 1, 2]).astype(np.int32))
    test_dynamic.test_dynamic_grad_net([x, segment_ids])


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dynamic_segment_mean_rank():
    """
    Feature: SegmentMean Grad DynamicRank.
    Description: Test case of dynamic rank for SegmentMean grad operator on CPU, GPU and Ascend.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    test_dynamic = TestDynamicGrad(NetSegmentMean())
    x = Tensor(np.array([2, 2, 3, 4]).astype(np.float32))
    segment_ids = Tensor(np.array([0, 0, 1, 2]).astype(np.int32))
    test_dynamic.test_dynamic_grad_net([x, segment_ids], True)

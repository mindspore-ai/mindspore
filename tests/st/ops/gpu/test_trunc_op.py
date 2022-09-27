import numpy as np
import pytest

import mindspore.context as context
from mindspore import Tensor
import mindspore.ops.operations as op
from mindspore import dtype

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_trunc_fp32():
    """
    Feature: Trunc Gpu  kernel.
    Description: test the trunc.
    Expectation: success.
    """
    x = Tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype.float32)
    net = op.Trunc()
    output = net(x)
    expect = Tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype.float32)
    assert np.allclose(output.asnumpy(), expect.asnumpy().astype(np.float32), 0.0001, 0.0001)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_trunc_fp16():
    """
    Feature: Gpu Trunc kernel.
    Description: test the trunc.
    Expectation: success.
    """
    x = Tensor(np.arange(12).reshape((3, 4)).astype(np.float16))
    net = op.Trunc()
    output = net(x)
    expect = np.arange(12).reshape((3, 4)).astype(np.float16)
    assert np.allclose(output.asnumpy(), expect, 0.0001, 0.0001)

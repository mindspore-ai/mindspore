import numpy as np
import pytest

import mindspore.context as context
from mindspore import Tensor
from mindspore import numpy as  P
from mindspore import dtype

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_trunc_fp32():
    """
    Feature: Trunc Gpu  kernel.
    Description: test the trunc.
    Expectation: success.
    """
    x = np.random.uniform(-1, 1, size=(3, 4)).astype(np.float32)
    output = P.trunc(Tensor(x, dtype=dtype.float32))
    expect = np.trunc(x)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_trunc_fp16():
    """
    Feature: Gpu Trunc kernel.
    Description: test the trunc.
    Expectation: success.
    """
    x = np.random.uniform(-1, 1, size=(3, 4)).astype(np.float16)
    output = P.trunc(Tensor(x, dtype=dtype.float16))
    expect = np.trunc(x)
    assert np.allclose(output.asnumpy(), expect)

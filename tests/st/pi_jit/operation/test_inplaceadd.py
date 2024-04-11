import numpy as np
import pytest
from ..share.ops.primitive.inplaceadd_ops import InplaceAddFactory


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_p_inplaceadd_input_3d_all():
    """
    Feature: Ops.
    Description: test operator InplaceAdd, given (inputx_shape=(128, 32),inputv_shape=(2, 32),dtype=np.float32).
    Expectation: expect correct result.
    """
    fact = InplaceAddFactory(inputx_shape=(8, 128, 64), inputv_shape=(8, 128, 64),
                             indices=(0, 1, 2, 3, 4, 5, 6, 7), dtype1=np.float32,
                             dtype2=np.float32)
    fact.forward_cmp()


@pytest.mark.level2
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_p_inplaceadd_input_3d_indices_int():
    """
    Feature: Ops.
    Description: test operator InplaceAdd, given (inputx_shape=(128, 32),inputv_shape=(2, 32),dtype=np.float32).
    Expectation: expect correct result.
    """
    fact = InplaceAddFactory(inputx_shape=(32, 128, 64), inputv_shape=(1, 128, 64), indices=18,
                             dtype1=np.float32, dtype2=np.float32)
    fact.forward_cmp()


@pytest.mark.level2
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_p_inplaceadd_input_4d_indices_all():
    """
    Feature: Ops.
    Description: test operator InplaceAdd, given (inputx_shape=(128, 32),inputv_shape=(2, 32),dtype=np.float32).
    Expectation: expect correct result.
    """
    fact = InplaceAddFactory(inputx_shape=(8, 128, 64, 2), inputv_shape=(8, 128, 64, 2),
                             indices=(0, 1, 2, 3, 4, 5, 6, 7), dtype1=np.float32,
                             dtype2=np.float32)
    fact.forward_cmp()


@pytest.mark.level3
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_p_inplaceadd_input_5d_7d():
    """
    Feature: Ops.
    Description: test operator InplaceAdd, given (inputx_shape=5d-7d).
    Expectation: expect correct result.
    """
    fact = InplaceAddFactory(inputx_shape=(16, 8, 8, 4, 4), inputv_shape=(4, 8, 8, 4, 4),
                             indices=(0, 1, 2, 3), dtype1=np.float32, dtype2=np.float32)
    fact.forward_cmp()

    fact = InplaceAddFactory(inputx_shape=(16, 8, 8, 4, 4, 2), inputv_shape=(4, 8, 8, 4, 4, 2),
                             indices=(0, 1, 14, 15), dtype1=np.float16, dtype2=np.float16)
    fact.forward_cmp()

    fact = InplaceAddFactory(inputx_shape=(16, 8, 8, 4, 4, 2, 2),
                             inputv_shape=(4, 8, 8, 4, 4, 2, 2), indices=(12, 13, 14, 15),
                             dtype1=np.float64, dtype2=np.float64)


@pytest.mark.level2
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_p_inplaceadd_input_1d():
    """
    Feature: Ops.
    Description: test operator InplaceAdd, given (inputx_shape=1d).
    Expectation: expect correct result.
    """
    fact = InplaceAddFactory(inputx_shape=(16,), inputv_shape=(4,), indices=(0, 1, 2, 3),
                             dtype1=np.float32, dtype2=np.float32)
    fact.forward_cmp()


@pytest.mark.level2
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_p_inplaceadd_indices_repeat():
    """
    Feature: Ops.
    Description: test operator InplaceAdd, given v repeat.
    Expectation: expect correct result.
    """
    fact = InplaceAddFactory(inputx_shape=(16, 8, 8, 4, 4), inputv_shape=(2, 8, 8, 4, 4),
                             indices=(1, 1), dtype1=np.float32, dtype2=np.float32)
    fact.forward_cmp()

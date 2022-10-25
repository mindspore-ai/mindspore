from array import array
import numpy as np
import pytest
import mindspore.common.dtype as mstype
from mindspore import context, Tensor


context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_multinomial():
    """
    Feature: multinomial
    Description: test the function of multinomial.
    Expectation: success
    """
    input_tensor = Tensor([0., 9., 4., 0.], mstype.float32)
    output = input_tensor.multinomial(num_samples=2, seed=10)
    expect_res = np.array([2, 1])
    assert np.allclose(output.asnumpy(), expect_res)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_rand_like():
    """
    Feature: rand_like
    Description: test the function of rand_like.
    Expectation: success
    """
    input_x = Tensor(np.array([[1, 2, 3, 9], [1, 2, 3, 9]]), mstype.int8)
    output = input_x.rand_like(seed=0)
    expect_res = np.array([[5.48813504e-01, 7.15189366e-01, 6.02763376e-01, 5.44883183e-01],
                           [4.23654799e-01, 6.45894113e-01, 4.37587211e-01, 8.91773001e-01]]).astype(np.float64)
    assert np.allclose(output.asnumpy(), expect_res)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_randint_like():
    """
    Feature: randint_like
    Description: test the function of randint_like.
    Expectation: success
    """
    input_x = Tensor(np.array([1., 2., 3., 4., 5.]), mstype.float32)
    output = input_x.randint_like(20, 100, seed=0)
    expect_res = np.array([64, 67, 84, 87, 87])
    assert np.allclose(output.asnumpy(), expect_res)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_randn_like():
    """
    Feature: randn_like
    Description: test the function of randn_like.
    Expectation: success
    """
    input_p = Tensor(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), mstype.int32)
    output = input_p.randn_like(seed=0)
    expect_res = np.array([[1.76405239e+00, 4.00157213e-01, 9.78738010e-01, 2.24089313e+00, 1.86755800e+00],
                           [-9.77277875e-01, 9.50088441e-01, -1.51357204e-01, -1.03218853e-01, 4.10598516e-01]])
    assert np.allclose(output.asnumpy(), expect_res)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_as_tensor():
    """
    Feature: as_tensor
    Description: test the function of as_tensor.
    Expectation: success
    """
    input_data = np.array([1, 2, 3])
    ms_tensor = Tensor.as_tensor(input_data)
    expect_res = np.array([1, 2, 3])
    assert np.allclose(ms_tensor.asnumpy(), expect_res)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_as_strided():
    """
    Feature: rand_like
    Description: test the function of as_stride.
    Expectation: success
    """
    input_array = np.arange(9, dtype=np.int32).reshape(3, 3)
    output = Tensor(input_array).as_strided((2, 2), (1, 1))
    expect_res = np.array([[0, 1], [1, 2]]).astype(np.int32)
    assert np.allclose(output.asnumpy(), expect_res)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_frombuffer():
    """
    Feature: rand_like
    Description: test the function of frombuffer.
    Expectation: success
    """
    input_array = array("d", [1, 2, 3, 4])
    output = Tensor.frombuffer(input_array, mstype.int32)
    expect_res = np.array([1, 2, 3, 4]).astype(np.int32)
    assert np.allclose(output.asnumpy(), expect_res)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_empty_strided():
    """
    Feature: rand_like
    Description: test the function of empty_strided.
    Expectation: success
    """
    size = (3, 3)
    stride = (1, 3)
    output = Tensor.empty_strided(size, stride, seed=0)
    expect_res = np.array([[0.00000000e+00, 7.15189366e+10, 0.00000000e+00],
                           [0.00000000e+00, 0.00000000e+00, 6.45894113e+10],
                           [0.00000000e+00, 8.91773001e+10, 9.63662761e+10]]).astype(np.float64)
    assert np.allclose(output.asnumpy(), expect_res)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_randpermn():
    """
    Feature: rand_like
    Description: test the function of randpermn.
    Expectation: success
    """
    n = Tensor([20], dtype=mstype.int32)
    output = n.randperm(max_length=30, pad=-1)
    set_a = set(np.linspace(0, 19, 20).astype(int))
    set_b = set(output.asnumpy())
    assert set_a.issubset(set_b)

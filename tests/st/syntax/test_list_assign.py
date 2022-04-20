import pytest
from mindspore.nn import Cell

from mindspore import Tensor
from mindspore import context


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_pynative_list_slice_tensor_no_step():
    """
    Feature: List assign
    Description: Test list slice assign with tensor
    Expectation: No exception.
    """

    class NetInner(Cell):
        def construct(self, start=None, stop=None, step=None):
            a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            b = Tensor([11, 22, 33])
            a[start:stop:step] = b
            return tuple(a)

    context.set_context(mode=context.PYNATIVE_MODE)
    net = NetInner()
    python_out = (Tensor(11), Tensor(22), Tensor(33), 4, 5, 6, 7, 8, 9)
    pynative_out = net(0, 3, None)
    assert pynative_out == python_out

    context.set_context(mode=context.GRAPH_MODE)
    graph_out = net(0, 3, None)
    assert graph_out == python_out


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_pynative_list_slice_tensor_with_step():
    """
    Feature: List assign
    Description: Test list slice assign with tensor
    Expectation: No exception.
    """

    class NetInner(Cell):
        def construct(self, start=None, stop=None, step=None):
            a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            b = Tensor([11, 22, 33])
            a[start:stop:step] = b
            return tuple(a)

    context.set_context(mode=context.PYNATIVE_MODE)
    net = NetInner()
    python_out = (Tensor(11), 2, 3, Tensor(22), 5, 6, Tensor(33), 8, 9)
    pynative_out = net(0, None, 3)
    assert python_out == pynative_out

    context.set_context(mode=context.GRAPH_MODE)
    graph_out = net(0, None, 3)
    assert python_out == graph_out

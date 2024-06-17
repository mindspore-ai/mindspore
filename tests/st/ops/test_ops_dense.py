
import numpy as np
import pytest

from mindspore.ops import Dense
from mindspore.ops.composite.base import GradOperation

import mindspore
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


class DenseCell(mindspore.nn.Cell):
    def __init__(self):
        super(DenseCell, self).__init__()
        self.dense = Dense()

    def construct(self, x, w, b):
        return self.dense(x, w, b)


def get_bias_shape(shape_w):
    if len(shape_w) == 2:
        return (shape_w[0],)
    return ()


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize(
    "shape_x, shape_w, has_bias, e_shape, e_grad_x_shape, e_grad_w_shape, e_grad_b_shape",
    [
        ((4, 2), (3, 2), False, (4, 3), (4, 2), (3, 2), None),
        ((4, 2), (3, 2), True, (4, 3), (4, 2), (3, 2), (3,)),
        ((4, 4, 2), (3, 2), False, (4, 4, 3), (4, 4, 2), (3, 2), None),
        ((4, 4, 2), (3, 2), True, (4, 4, 3), (4, 4, 2), (3, 2), (3,)),
        ((2,), (3, 2), False, (3,), (2,), (3, 2), None),
        ((2,), (3, 2), True, (3,), (2,), (3, 2), (3,)),
        ((2,), (2,), False, (), (2,), (2,), None),
        ((2,), (2,), True, (), (2,), (2,), ()),
        ((3, 2,), (2,), False, (3,), (3, 2,), (2,), None),
        ((4, 3, 2,), (2,), False, (4, 3,), (4, 3, 2,), (2,), None),
    ],
)
@pytest.mark.parametrize(
    "mode, dynamic",
    [
        [mindspore.PYNATIVE_MODE, 0],
        [mindspore.GRAPH_MODE, 0],
    ],
)
def test_static(mode, dynamic, shape_x, shape_w, has_bias, e_shape, e_grad_x_shape, e_grad_w_shape, e_grad_b_shape):
    """
    Feature: ops.Dense
    Description: static
    Expectation: success
    """
    dense_case(dynamic, e_grad_b_shape, e_grad_w_shape, e_grad_x_shape, e_shape, has_bias, mode, shape_w, shape_x)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize(
    "shape_x, shape_w, has_bias, e_shape, e_grad_x_shape, e_grad_w_shape, e_grad_b_shape",
    [
        ((4, 2), (3, 2), False, (4, 3), (4, 2), (3, 2), None),
        ((4, 2), (3, 2), True, (4, 3), (4, 2), (3, 2), (3,)),
        ((4, 4, 2), (3, 2), False, (4, 4, 3), (4, 4, 2), (3, 2), None),
        ((4, 4, 2), (3, 2), True, (4, 4, 3), (4, 4, 2), (3, 2), (3,)),
        ((2,), (3, 2), False, (3,), (2,), (3, 2), None),
        ((2,), (3, 2), True, (3,), (2,), (3, 2), (3,)),
        ((2,), (2,), False, (), (2,), (2,), None),
        ((2,), (2,), True, (), (2,), (2,), ()),
        ((3, 2,), (2,), False, (3,), (3, 2,), (2,), None),
        ((4, 3, 2,), (2,), False, (4, 3,), (4, 3, 2,), (2,), None),
    ],
)
@pytest.mark.parametrize(
    "mode, dynamic",
    [
        [mindspore.GRAPH_MODE, 1],
        [mindspore.GRAPH_MODE, 2],
        [mindspore.GRAPH_MODE, 3],
        [mindspore.GRAPH_MODE, 4],
    ],
)
def test_dynamic(mode, dynamic, shape_x, shape_w, has_bias, e_shape, e_grad_x_shape, e_grad_w_shape, e_grad_b_shape):
    """
    Feature: ops.Dense
    Description: dynamic
    Expectation: success
    """
    dense_case(dynamic, e_grad_b_shape, e_grad_w_shape, e_grad_x_shape, e_shape, has_bias, mode, shape_w, shape_x)


def dense_case(dynamic, e_grad_b_shape, e_grad_w_shape, e_grad_x_shape, e_shape, has_bias, mode, shape_w, shape_x):
    mindspore.set_context(jit_level='O0')
    dense_cell = DenseCell()
    mindspore.context.set_context(mode=mode)
    x = random_input(shape_x)
    w = random_input(shape_w)
    b = random_input(get_bias_shape(shape_w)) if has_bias else None
    x_ms = mindspore.tensor(x)
    w_ms = mindspore.tensor(w)
    b_ms = mindspore.tensor(b) if has_bias else None
    x_rank = len(shape_x)
    w_rank = len(shape_w)
    b_rank = len(get_bias_shape(shape_w))
    if dynamic == 1:

        if b_rank == 0:
            bias = mindspore.tensor(b, dtype=mindspore.float32)
        else:
            bias = mindspore.tensor(
                shape=[None] * b_rank, dtype=mindspore.float32)

        dense_cell.set_inputs(
            mindspore.tensor(shape=[None] * x_rank, dtype=mindspore.float32),
            mindspore.tensor(shape=[None] * w_rank, dtype=mindspore.float32),
            bias if has_bias else None,
        )
    elif dynamic == 2:
        dense_cell.set_inputs(
            mindspore.tensor(shape=None, dtype=mindspore.float32),
            mindspore.tensor(shape=None, dtype=mindspore.float32),
            mindspore.tensor(
                shape=None, dtype=mindspore.float32) if has_bias else None,
        )
    elif dynamic == 3:
        dense_cell.set_inputs(
            mindspore.tensor(shape=[None] * x_rank, dtype=mindspore.float32),
            mindspore.tensor(shape=None, dtype=mindspore.float32),
            mindspore.tensor(
                shape=None, dtype=mindspore.float32) if has_bias else None,
        )
    elif dynamic == 4:
        dense_cell.set_inputs(
            mindspore.tensor(shape=None, dtype=mindspore.float32),
            mindspore.tensor(shape=[None] * w_rank, dtype=mindspore.float32),
            mindspore.tensor(
                shape=None, dtype=mindspore.float32) if has_bias else None,
        )
    actual = dense_cell(x_ms, w_ms, b_ms)
    assert actual.shape == e_shape
    dense_cell_grad = GradOperation(get_all=True, sens_param=True)(dense_cell)
    if has_bias:
        actual_grad_x, actual_grad_w, actual_grad_b = dense_cell_grad(
            x_ms, w_ms, b_ms, actual
        )
    else:
        actual_grad_x, actual_grad_w = dense_cell_grad(
            x_ms, w_ms, b_ms, actual)
    assert actual_grad_x.shape == e_grad_x_shape
    assert actual_grad_w.shape == e_grad_w_shape
    if has_bias:
        assert actual_grad_b.shape == e_grad_b_shape


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
def test_op():
    """
    Feature: ops.Dense
    Description: TEST_OP
    Expectation: success
    """
    dense_cell = DenseCell()
    inputs_seq = []
    for shape_x, shape_w in [[(4, 2), (3, 2)], [(4, 4, 2), (3, 2)]]:
        x = mindspore.tensor(random_input(shape_x))
        w = mindspore.tensor(random_input(shape_w))
        b = mindspore.tensor(random_input(get_bias_shape(shape_w)))

        inputs_seq.append([x, w, b])

    TEST_OP(dense_cell, inputs_seq, '', disable_input_check=True, disable_yaml_check=True)


def random_input(shape, dtype=np.float32):
    return np.array(np.random.randn(*shape), dtype=dtype)

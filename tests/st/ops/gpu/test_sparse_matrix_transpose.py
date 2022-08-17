import random
import numpy as np
from mindspore.common.tensor import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P


class Net(Cell):
    def __init__(self, conjugate=False):
        super(Net, self).__init__()
        self.sparse_matrix_transpose = P.SparseMatrixTranspose(conjugate=conjugate)

    def construct(self, x_dense_shape, x_batch_ptrs, x_row_ptrs, x_col_inds, x_values):
        output = self.sparse_matrix_transpose(x_dense_shape, x_batch_ptrs, x_row_ptrs, x_col_inds, x_values)
        return output


def to_csr(x, index_type=np.int32, value_type=np.float32):
    x_dense_shape = x.shape
    batches, rows, cols = 1, x_dense_shape[0], x_dense_shape[1]
    if len(x_dense_shape) == 3:
        batches, rows, cols = x_dense_shape[0], x_dense_shape[1], x_dense_shape[2]
    x_batch_ptrs = np.zeros((batches+1))
    x_row_ptrs = []
    x_col_inds = []
    x_values = []
    values = x.flatten()
    value_cnt = rows*cols
    for i in range(batches):
        start_pos = i*(rows+1)
        for j in range(rows+1):
            x_row_ptrs.append(0)
        for j in range(value_cnt):
            if not values[value_cnt*i+j] == 0:
                row_idx = j//cols
                col_idx = j % cols
                x_batch_ptrs[i+1] += 1
                x_row_ptrs[start_pos+row_idx+1] += 1
                x_col_inds.append(col_idx)
                x_values.append(values[value_cnt*i+j])
        for j in range(rows):
            x_row_ptrs[j+start_pos+1] += x_row_ptrs[j+start_pos]
        x_batch_ptrs[i+1] += x_batch_ptrs[i]
    x_dense_shape = np.array(x_dense_shape).astype(index_type)
    x_batch_ptrs = np.array(x_batch_ptrs).astype(index_type)
    x_row_ptrs = np.array(x_row_ptrs).astype(index_type)
    x_col_inds = np.array(x_col_inds).astype(index_type)
    x_values = np.array(x_values).astype(value_type)
    res = tuple(x_dense_shape, x_batch_ptrs, x_row_ptrs, x_col_inds, x_values)
    return res


def test_1():
    """
    Feature: SparseMatrixTranspose gpu routine
    Description: common input
    Expectation: success
    """
    x = np.random.randint(low=0, high=10, size=(2, 3, 3))
    res = to_csr(x)
    x_dense_shape, x_batch_ptrs, x_row_ptrs, x_col_inds, x_values = (
        Tensor(i) for i in res)
    net = Net(False)
    x_dense_shape, x_batch_ptrs, x_row_ptrs, x_col_inds = x_dense_shape.astype(
        np.int64), x_batch_ptrs.astype(np.int64), x_row_ptrs.astype(np.int64), x_col_inds.astype(np.int64)
    output = net(x_dense_shape, x_batch_ptrs, x_row_ptrs, x_col_inds, x_values)
    print(output)


def test_2():
    """
    Feature: SparseMatrixTranspose gpu routine
    Description: common input
    Expectation: success
    """
    x = np.random.randint(low=0, high=10, size=(2, 3, 3))
    p = x.flatten().astype(np.complex64)
    ll = len(p)
    for i in range(ll):
        p[i] = complex(random.randint(1, 10), random.randint(1, 10))
    x = p.reshape(x.shape)
    res = to_csr(x, value_type=np.complex64)
    x_dense_shape, x_batch_ptrs, x_row_ptrs, x_col_inds, x_values = (
        Tensor(i) for i in res)
    x_dense_shape, x_batch_ptrs, x_row_ptrs, x_col_inds = x_dense_shape.astype(
        np.int64), x_batch_ptrs.astype(np.int64), x_row_ptrs.astype(np.int64), x_col_inds.astype(np.int64)
    net = Net(conjugate=True)
    output = net(x_dense_shape, x_batch_ptrs, x_row_ptrs, x_col_inds, x_values)
    print(output)

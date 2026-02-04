# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import numpy as np
import scipy.sparse
import mindspore as ms
from mindspore.ops.operations._csr_ops import CSRReduceSum


class GraphDataset:
    """Full training numpy dataset """
    def __init__(self, data_path="") -> None:
        if data_path:
            npz = np.load(data_path)
            self.n_nodes = npz.get('n_nodes', default=npz['feat'].shape[0])
            self.n_edges = npz.get('n_edges', default=npz['adj_coo_row'].shape[0])
            self.n_classes = int(npz.get('n_classes', default=np.max(npz['label']) + 1))
            row_indices = np.asarray(npz["adj_coo_row"], dtype=np.int32)
            col_indices = np.asarray(npz["adj_coo_col"], dtype=np.int32)
            out_deg = np.bincount(row_indices, minlength=self.n_nodes)
            in_deg = np.bincount(col_indices, minlength=self.n_nodes)
            in_deg = npz.get('in_degrees', default=in_deg)
            out_deg = npz.get('out_degrees', default=out_deg)
        else:
            self.n_nodes = 198020
            self.n_classes = 41
            self.n_edges = 84120742
            indptr_choice = np.arange(0, self.n_edges, dtype=np.int32)
            indptr = np.sort(np.random.choice(indptr_choice, self.n_nodes - 1, replace=True))
            indptr = np.concatenate(
                (np.array([0], dtype=np.int32), indptr, np.array([self.n_edges], dtype=np.int32)))
            indices_choice = np.arange(self.n_nodes, dtype=np.int32)
            idx_range = indices_choice.copy()
            idx_bound = np.diff(indptr).reshape(-1, 1)
            max_count = idx_bound.max().tolist()
            np.random.shuffle(indices_choice)
            indices_choice = indices_choice[: max_count]
            indices_choice.sort()
            indices = np.where(idx_range[: max_count] < idx_bound, indices_choice, self.n_nodes)
            mask = np.less(indices, self.n_nodes).nonzero()
            indices = indices[mask]
            csr_tensor = scipy.sparse.csr_matrix((np.ones(self.n_edges), indices, indptr),
                                                 shape=(self.n_nodes, self.n_nodes))
            coo_tensor = csr_tensor.tocoo()
            row_indices = np.asarray(coo_tensor.row, dtype=np.int32)
            col_indices = np.asarray(coo_tensor.col, dtype=np.int32)
            out_deg = np.bincount(row_indices, minlength=self.n_nodes)
            in_deg = np.bincount(col_indices, minlength=self.n_nodes)

        side = int(self.n_nodes)
        nnz = row_indices.shape[0]
        idx_forward = np.argsort(out_deg)[::-1]
        arg_idx_forward = np.argsort(idx_forward)

        row_indices_forward = arg_idx_forward[row_indices]
        col_indices_forward = arg_idx_forward[col_indices]
        coo_tensor_forward = scipy.sparse.coo_matrix(
            (np.ones(nnz), (row_indices_forward, col_indices_forward)), shape=(side, side))
        csr_tensor_forward = coo_tensor_forward.tocsr()

        self.row_indices = ms.Tensor(np.sort(row_indices_forward), dtype=ms.int32)
        self.indptr = ms.Tensor(np.asarray(csr_tensor_forward.indptr), dtype=ms.int32)
        self.indices = ms.Tensor(np.asarray(csr_tensor_forward.indices), dtype=ms.int32)

        coo_tensor_backward = scipy.sparse.csr_matrix(
            (np.ones(nnz), (col_indices_forward, row_indices_forward)), shape=(side, side))
        csr_tensor_backward = coo_tensor_backward.tocsr()

        self.indptr_backward = ms.Tensor(np.asarray(csr_tensor_backward.indptr), dtype=ms.int32)
        self.indices_backward = ms.Tensor(np.asarray(csr_tensor_backward.indices), dtype=ms.int32)

        if data_path:
            self.x = ms.Tensor(npz['feat'][idx_forward])
            self.y = ms.Tensor(npz['label'][idx_forward], ms.int32)
            self.train_mask = npz.get('train_mask', default=None)
            if self.train_mask is not None:
                self.train_mask = self.train_mask[idx_forward]
            self.test_mask = npz.get('test_mask', default=None)
            if self.test_mask is not None:
                self.test_mask = self.test_mask[idx_forward]
        else:
            self.x = ms.Tensor(np.random.rand(198020, 602), dtype=ms.float32)
            self.y = ms.Tensor(np.random.randint(0, high=198020, size=198020), ms.int32)
            self.train_mask = ms.Tensor(np.ones(self.n_nodes), dtype=ms.float32)
            self.test_mask = None
        self.in_deg = ms.Tensor(in_deg[idx_forward], ms.int32)
        self.out_deg = ms.Tensor(out_deg[idx_forward], ms.int32)


class GatherNet(ms.nn.Cell):
    def __init__(self, indptr_backward, indices_backward):
        super().__init__()
        self.indptr_backward = indptr_backward
        self.indices_backward = indices_backward

    def construct(self, data, indices, axis):
        return ms.ops.gather(data, indices, axis)

    def bprop(self, data, indices, axis, out, dout):
        grad_csr = ms.CSRTensor(
            self.indptr_backward, self.indices_backward, dout, (data.shape[0], data.shape[0]) + data.shape[1:])
        grad_sum = grad_csr.sum(1).reshape(data.shape)
        return (grad_sum, (), 0)


class CSRReduceSumNet(ms.nn.Cell):
    def __init__(self, indices_backward):
        super().__init__()
        self.indices_backward = indices_backward
        self.op = CSRReduceSum()

    def construct(self, indptr, indices, values, shape, axis):
        return self.op(indptr, indices, values, shape, axis)

    def bprop(self, indptr, indices, values, shape, axis, out, dout):
        dout = dout.reshape((dout.shape[0],) + dout.shape[2:])
        grad_values = ms.ops.gather(dout, self.indices_backward, 0)
        return (indptr, indices, grad_values, (), 0)

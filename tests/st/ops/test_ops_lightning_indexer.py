# Copyright 2025 Huawei Technologies Co., Ltd
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
# pylint: disable=too-many-function-args
"""Test cases for lightning_indexer operator."""
import pytest
import torch
import hashlib
import numpy as np
import mindspore as ms
from mindspore import ops
from mindspore.common.api import _pynative_executor
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP

_env_has_torch_npu = True
try:
    import torch_npu
    from torch_npu import npu_lightning_indexer  # pylint: disable=unused-import
    torch.npu.set_device(f'npu:{ms.get_context("device_id")}')
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = False
    print("\n====>torch_npu is found")
except ImportError:
    print("\n====>torch_npu or torch_npu.npu_lightning_indexer not found, benchmark will use torch cpu instead")
    _env_has_torch_npu = False
torch.use_deterministic_algorithms(True)


@test_utils.run_with_cell
def lightning_indexer_forward_func(query, key, weights, actual_seq_lengths_query=None, actual_seq_lengths_key=None,
                                   block_table=None, layout_query='BSND', layout_key='BSND', sparse_count=2048,
                                   sparse_mode=3, pre_tokens=9223372036854775807, next_tokens=9223372036854775807,
                                   return_value=False):
    return ops.lightning_indexer(query, key, weights, actual_seq_lengths_query=actual_seq_lengths_query,
                                 actual_seq_lengths_key=actual_seq_lengths_key, block_table=block_table,
                                 layout_query=layout_query, layout_key=layout_key, sparse_count=sparse_count,
                                 sparse_mode=sparse_mode, pre_tokens=pre_tokens, next_tokens=next_tokens,
                                 return_value=return_value)

@test_utils.run_with_cell
def lightning_indexer_backward_func(query, key, weights, actual_seq_lengths_query=None, actual_seq_lengths_key=None,
                                    block_table=None, layout_query='BSND', layout_key='BSND', sparse_count=2048,
                                    sparse_mode=3, pre_tokens=9223372036854775807, next_tokens=9223372036854775807,
                                    return_value=False):
    return ms.grad(lightning_indexer_forward_func, (0, 1, 2))(query, key, weights, actual_seq_lengths_query,
                                                              actual_seq_lengths_key, block_table, layout_query,
                                                              layout_key, sparse_count, sparse_mode, pre_tokens,
                                                              next_tokens, return_value)

def torch_npu_lightning_indexer_forward_backward(query, key, weights, actual_seq_lengths_query=None,
                                                 actual_seq_lengths_key=None, block_table=None, layout_query='BSND',
                                                 layout_key='BSND', sparse_count=2048, sparse_mode=3,
                                                 pre_tokens=9223372036854775807, next_tokens=9223372036854775807,
                                                 return_value=False, grad=False):
    query = query.npu()
    key = key.npu()
    weights = weights.npu()
    if actual_seq_lengths_query is not None:
        actual_seq_lengths_query = actual_seq_lengths_query.npu()
    if actual_seq_lengths_key is not None:
        actual_seq_lengths_key = actual_seq_lengths_key.npu()
    if block_table is not None:
        block_table = block_table.npu()

    if grad:
        query.requires_grad = True
        key.requires_grad = True
        weights.requires_grad = True

    sparse_indices, sparse_values = torch_npu.npu_lightning_indexer(
        query, key, weights, actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=actual_seq_lengths_key, block_table=block_table, layout_query=layout_query,
        layout_key=layout_key, sparse_count=sparse_count, sparse_mode=sparse_mode, pre_tokens=pre_tokens,
        next_tokens=next_tokens, return_value=return_value)

    if not grad:
        return sparse_indices.detach(), sparse_values.detach()
    sparse_values.backward(gradient=torch.ones_like(sparse_values))
    return sparse_indices.detach(), sparse_values.detach(), query.grad.detach(), key.grad.detach(), \
        weights.grad.detach()

class TorchCpuLightningIndexer():
    def __init__(self, query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key, block_table, layout_query,
                 layout_key, sparse_count, sparse_mode, return_value):
        self.query = query
        self.key = key
        self.weights = weights
        self.actual_seq_lengths_query = actual_seq_lengths_query
        self.actual_seq_lengths_key = actual_seq_lengths_key
        self.block_table = block_table
        self.layout_query = layout_query
        self.layout_key = layout_key
        self.sparse_count = sparse_count
        self.sparse_mode = sparse_mode
        self.return_value = return_value

        # Determine dimensions based on layout
        if layout_query == 'BSND':
            self.B, self.S1, self.N1, self.D = query.shape
        else:  # TND
            T1, self.N1, self.D = query.shape
            self.B = actual_seq_lengths_query.shape[0]
            self.S1 = T1  # Use total tokens as S1 for TND

        if layout_key == 'PA_BSND':
            block_count, block_size, self.N2, _ = key.shape
            self.S2 = block_count * block_size // self.B if block_table is not None else block_count * block_size
        elif layout_key == 'TND':
            T2, self.N2, _ = key.shape
            self.S2 = T2
        else:  # BSND
            _, self.S2, self.N2, _ = key.shape

        self.group_num = self.N1 // self.N2

    def _get_data_from_pa_cache(self, key, block_table, act_s2):
        _, block_size, N2, D = key.shape
        if N2 != 1:
            raise ValueError("N2 only support 1")
        need_blcok_num = (act_s2 + block_size - 1) // block_size
        act_s2_align = need_blcok_num * block_size
        out = torch.zeros((act_s2_align, D), dtype=key.dtype, device=key.device)
        for i in range(need_blcok_num):
            out[i * block_size: (i+1) * block_size, :] = key[block_table[i], ...].reshape(block_size, D)

        return out[:act_s2, :]

    def _safe_mm(self, a, b):
        """
        Safe matrix multiplication that avoids torch.matmul bug on AArch64.

        On AArch64 with OneDNN+ACL, torch.matmul fails when the second matrix has
        only 1 column (out_features=1). This function uses torch.mv as a workaround.
        See: https://github.com/pytorch/pytorch/issues/110149

        Args:
            a: First matrix (m, k)
            b: Second matrix (k, n)

        Returns:
            Matrix product (m, n)
        """
        if b.shape[-1] == 1:
            # Use mv + unsqueeze to avoid the bug: (m, k) @ (k, 1) -> (m, 1)
            return torch.mv(a, b.squeeze(-1)).unsqueeze(-1)
        return torch.mm(a, b)

    def forward(self):
        """
        Compute Top-k sparse indices based on DSA algorithm.

        Algorithm (per query token):
            Indices = Top-k { [1]_{1xg} @ [(W @ [1]_{1xSk}) ⊙ ReLU(Q @ K^T)] }

        where:
            Q ∈ R^{gxd} : query for current token (N1 heads, D dim)
            K ∈ R^{Skxd}: context key (S2 tokens, D dim)
            W ∈ R^{gx1} : head weights (N1 heads)
            g = N1 (GQA group size / number of query heads)
            d = D (head dimension, 128)
            Sk = S2 (context length)
        """
        # Extract inputs from self
        query = self.query
        key = self.key
        weights = self.weights
        actual_seq_lengths_query = self.actual_seq_lengths_query
        actual_seq_lengths_key = self.actual_seq_lengths_key
        block_table = self.block_table
        layout_query = self.layout_query
        layout_key = self.layout_key
        sparse_count = self.sparse_count
        sparse_mode = self.sparse_mode
        return_value = self.return_value

        # Determine dimensions based on layout
        if layout_query == 'BSND':
            B, S1, _, _ = query.shape
        else:  # TND
            _, _, _ = query.shape
            B = actual_seq_lengths_query.shape[0]
            S1 = None  # Will be determined per batch

        if layout_key == 'PA_BSND':
            _, _, N2, _ = key.shape
        elif layout_key == 'TND':
            _, N2, _ = key.shape
        else:  # BSND
            _, S2_max, N2, _ = key.shape

        # Initialize output tensor
        out_shape = list(query.shape)
        out_shape[-1] = sparse_count
        out_shape[-2] = N2
        sparse_indices = torch.full(out_shape, -1, dtype=torch.int32, device=query.device)
        sparse_values = torch.zeros(out_shape, dtype=query.dtype, device=query.device) if return_value else None

        # Flatten output for easier indexing
        sparse_indices_flat = sparse_indices.reshape(-1, N2, sparse_count)
        sparse_values_flat = sparse_values.reshape(-1, N2, sparse_count) if return_value else None

        # Process each batch
        q_offset = 0  # Cumulative offset for query in TND layout
        k_offset = 0  # Cumulative offset for key in TND layout

        for batch_idx in range(B):
            # Get actual sequence lengths for current batch
            if layout_query == 'TND':
                curr_s1 = (actual_seq_lengths_query[batch_idx] - q_offset).item()
            elif actual_seq_lengths_query is not None:
                curr_s1 = actual_seq_lengths_query[batch_idx].item()
            else:
                curr_s1 = S1

            if layout_key == 'TND':
                curr_s2 = (actual_seq_lengths_key[batch_idx] - k_offset).item()
            elif actual_seq_lengths_key is not None:
                curr_s2 = actual_seq_lengths_key[batch_idx].item()
            elif layout_key == 'PA_BSND':
                raise ValueError("actual_seq_lengths_key is required for PA_BSND layout")
            else:
                curr_s2 = S2_max

            # Extract query for current batch: (curr_s1, N1, D)
            if layout_query == 'BSND':
                q_batch = query[batch_idx, :curr_s1, :, :]  # (S1, N1, D)
                w_batch = weights[batch_idx, :curr_s1, :]   # (S1, N1)
            else:  # TND
                q_batch = query[q_offset:q_offset + curr_s1, :, :]  # (S1, N1, D)
                w_batch = weights[q_offset:q_offset + curr_s1, :]   # (S1, N1)

            # Extract key for current batch: (curr_s2, D)
            if layout_key == 'PA_BSND':
                k_batch = self._get_data_from_pa_cache(key, block_table[batch_idx], curr_s2)  # (S2, D)
            elif layout_key == 'BSND':
                k_batch = key[batch_idx, :curr_s2, 0, :]  # (S2, D), N2=1
            else:  # TND
                k_batch = key[k_offset:k_offset + curr_s2, 0, :]  # (S2, D), N2=1

            # Convert to float32 for computation precision
            q_batch = q_batch.to(torch.float32)  # (S1, N1, D)
            k_batch = k_batch.to(torch.float32)  # (S2, D)
            w_batch = w_batch.to(torch.float32)  # (S1, N1)

            # Process each query token
            for s1_idx in range(curr_s1):
                # Q_index: (N1, D) - query vector for current token
                Q_index = q_batch[s1_idx, :, :]  # (N1, D) = (g, d)

                # K_index: (S2, D) - context key
                K_index = k_batch  # (Sk, d)

                # W: (N1, 1) - head weights for current token
                W = w_batch[s1_idx, :].unsqueeze(1)  # (g, 1)

                # Step 1: Q @ K^T -> (N1, S2) = (g, Sk)
                qk_scores = self._safe_mm(Q_index, K_index.T)  # (g, Sk)

                # Step 2: ReLU(Q @ K^T) -> (N1, S2)
                relu_scores = torch.clamp(qk_scores, min=0.0)  # (g, Sk)

                # Step 3: W @ [1]_{1xSk} -> (N1, S2) - broadcast weights
                # This is equivalent to: W.expand(-1, curr_s2)
                W_broadcast = W.expand(-1, curr_s2)  # (g, Sk)

                # Step 4: Hadamard product (element-wise multiply)
                weighted_scores = W_broadcast * relu_scores  # (g, Sk)

                # Step 5: [1]_{1xg} @ [...] -> (1, S2) - sum over heads
                # Equivalent to: sum over dim 0
                aggregated_scores = torch.sum(weighted_scores, dim=0)  # (Sk,)

                # Step 6: Apply causal mask if sparse_mode == 3 (rightDownCausal)
                if sparse_mode == 3:
                    # In rightDownCausal mode, for query at position s1_idx (from the end),
                    # mask out key positions beyond the causal boundary
                    # The last query token can attend to all keys
                    # The second-to-last can attend to all but the last key, etc.
                    causal_boundary = curr_s2 - (curr_s1 - 1 - s1_idx)
                    if causal_boundary < curr_s2:
                        aggregated_scores[causal_boundary:] = float('-inf')

                # Step 7: Top-k selection using sort (consistent with forward)
                k_select = min(sparse_count, curr_s2)
                sorted_values, sorted_indices = torch.sort(aggregated_scores, descending=True, stable=True)
                topk_indices = sorted_indices[:k_select]
                topk_values = sorted_values[:k_select]

                # Store results
                out_idx = q_offset + s1_idx
                sparse_indices_flat[out_idx, 0, :k_select] = topk_indices.to(torch.int32)
                if return_value:
                    sparse_values_flat[out_idx, 0, :k_select] = topk_values.to(query.dtype)

            # Update offsets for TND layout
            q_offset += curr_s1
            k_offset += curr_s2

        # Reshape back to original output shape
        sparse_indices = sparse_indices_flat.reshape(out_shape)
        if return_value:
            sparse_values = sparse_values_flat.reshape(out_shape)
            return sparse_indices, sparse_values
        return sparse_indices, None

    def _get_real_topk(self, s1_idx, curr_s1, curr_s2):
        """
        Get the real topk count based on sparse_mode and causal mask.

        For sparse_mode=0 (defaultMask): return sparse_count
        For sparse_mode=3 (rightDownCausal): return min(causal_boundary, sparse_count)
        """
        if self.sparse_mode == 0:
            return min(self.sparse_count, curr_s2)
        # sparse_mode == 3 (rightDownCausal)
        # In rightDownCausal mode, causal_boundary = s2 - (s1 - 1 - s1_idx)
        causal_boundary = curr_s2 - (curr_s1 - 1 - s1_idx)
        if causal_boundary <= 0:
            return 0
        return min(causal_boundary, self.sparse_count)

    def backward(self, sparse_indices, value_grad):
        """
        Compute gradients for LightningIndexer backward pass.

        Algorithm:
            Given forward: Indices = Top-k { [1]_{1xg} @ [(W @ [1]_{1xSk}) ⊙ ReLU(Q @ K^T)] }
            The backward computes dQ, dK, dW based on the chain rule.

        For each query token:
            1. mulGrad = W @ dV (weights multiply with value gradient)
            2. reluIn = Q @ gatheredK^T (recompute forward QK with selected keys)
            3. reluGrad = mulGrad * (reluIn > 0) (ReLU backward)
            4. dQ = reluGrad @ gatheredK (gradient w.r.t. query)
            5. dK = scatter_add(reluGrad^T @ Q) (gradient w.r.t. key with scatter)
            6. dW = sum(ReLU(reluIn) * dV, dim=-1) (gradient w.r.t. weights)

        Args:
            sparse_indices: Top-k indices from forward pass, shape (B, S1, N2, sparse_count) or (T, N2, sparse_count)
            value_grad: Gradient of sparse_values, shape same as sparse_indices

        Returns:
            dq: Gradient w.r.t. query, same shape as query
            dk: Gradient w.r.t. key, same shape as key
            dw: Gradient w.r.t. weights, same shape as weights
        """
        query = self.query
        key = self.key
        weights = self.weights
        actual_seq_lengths_query = self.actual_seq_lengths_query
        actual_seq_lengths_key = self.actual_seq_lengths_key
        block_table = self.block_table
        layout_query = self.layout_query
        layout_key = self.layout_key
        sparse_count = self.sparse_count
        _ = self.sparse_mode

        B = self.B
        N1 = self.N1
        N2 = self.N2
        D = self.D
        group_num = self.group_num

        # Determine dimensions based on layout
        S1 = None  # Initialize S1 to avoid possibly-used-before-assignment
        if layout_query == 'BSND':
            B, S1, N1, D = query.shape
        else:  # TND
            _, N1, D = query.shape

        if layout_key == 'PA_BSND':
            _, _, N2, _ = key.shape
        elif layout_key == 'TND':
            _, N2, _ = key.shape
        else:  # BSND
            _, S2_max, N2, _ = key.shape

        # Initialize output gradients
        dq = torch.zeros_like(query, dtype=torch.float32)
        if layout_key == 'PA_BSND':
            # For PA_BSND, we need to handle key shape differently
            dk_shape = key.shape
            dk = torch.zeros(dk_shape, dtype=torch.float32)
        else:
            dk = torch.zeros_like(key, dtype=torch.float32)
        dw = torch.zeros_like(weights, dtype=torch.float32)

        # Flatten for easier indexing
        sparse_indices_flat = sparse_indices.reshape(-1, N2, sparse_count)
        value_grad_flat = value_grad.reshape(-1, N2, sparse_count)

        # Process each batch
        q_offset = 0
        k_offset = 0

        for batch_idx in range(B):
            # Get actual sequence lengths for current batch
            if layout_query == 'TND':
                curr_s1 = (actual_seq_lengths_query[batch_idx] - q_offset).item()
            elif actual_seq_lengths_query is not None:
                curr_s1 = actual_seq_lengths_query[batch_idx].item()
            else:
                curr_s1 = S1

            if layout_key == 'TND':
                curr_s2 = (actual_seq_lengths_key[batch_idx] - k_offset).item()
            elif actual_seq_lengths_key is not None:
                curr_s2 = actual_seq_lengths_key[batch_idx].item()
            elif layout_key == 'PA_BSND':
                raise ValueError("actual_seq_lengths_key is required for PA_BSND layout")
            else:
                curr_s2 = S2_max

            # Extract data for current batch
            if layout_query == 'BSND':
                q_batch = query[batch_idx, :curr_s1, :, :]  # (S1, N1, D)
                w_batch = weights[batch_idx, :curr_s1, :]   # (S1, N1)
            else:  # TND
                q_batch = query[q_offset:q_offset + curr_s1, :, :]  # (S1, N1, D)
                w_batch = weights[q_offset:q_offset + curr_s1, :]   # (S1, N1)

            if layout_key == 'PA_BSND':
                k_batch = self._get_data_from_pa_cache(key, block_table[batch_idx], curr_s2)  # (S2, D)
            elif layout_key == 'BSND':
                k_batch = key[batch_idx, :curr_s2, 0, :]  # (S2, D), N2=1
            else:  # TND
                k_batch = key[k_offset:k_offset + curr_s2, 0, :]  # (S2, D), N2=1

            # Convert to float32 for computation precision
            q_batch = q_batch.to(torch.float32)  # (S1, N1, D)
            k_batch = k_batch.to(torch.float32)  # (S2, D)
            w_batch = w_batch.to(torch.float32)  # (S1, N1)

            # Initialize batch gradients
            dq_batch = torch.zeros((curr_s1, N1, D), dtype=torch.float32)
            dk_batch = torch.zeros((curr_s2, D), dtype=torch.float32)
            dw_batch = torch.zeros((curr_s1, N1), dtype=torch.float32)

            # Process each query token
            for s1_idx in range(curr_s1):
                out_idx = q_offset + s1_idx
                topk_indices = sparse_indices_flat[out_idx, 0, :]  # (sparse_count,)
                dv = value_grad_flat[out_idx, 0, :].to(torch.float32)  # (sparse_count,)

                # Get real topk count based on sparse_mode
                real_topk = self._get_real_topk(s1_idx, curr_s1, curr_s2)
                if real_topk == 0:
                    continue

                # Get valid indices and gradients
                valid_indices = topk_indices[:real_topk].long()  # (real_topk,)
                dv_valid = dv[:real_topk]  # (real_topk,)

                # Q_index: (N1, D) -> reshape to (N2, group_num, D) for GQA
                Q_index = q_batch[s1_idx, :, :].reshape(N2, group_num, D)  # (N2, g, D)
                Q_index = Q_index.squeeze(0)  # (g, D) since N2=1

                # W: (N1,) -> (N2, group_num) -> (g,)
                W = w_batch[s1_idx, :].reshape(N2, group_num).squeeze(0)  # (g,)

                # Gather selected keys: (real_topk, D)
                K_selected = k_batch[valid_indices, :]  # (real_topk, D)

                # Step 1: mulGrad = W @ dV -> (g, 1) * (1, real_topk) -> (g, real_topk)
                # Use broadcasting instead of matmul to avoid torch.matmul bug when second input is (1,1)
                # See: https://github.com/pytorch/pytorch/issues/110149
                W_col = W.unsqueeze(1)  # (g, 1)
                dv_row = dv_valid.unsqueeze(0)  # (1, real_topk)
                mulGrad = W_col * dv_row  # (g, real_topk) via broadcasting

                # Step 2: reluIn = Q @ K_selected^T -> (g, D) @ (D, real_topk) -> (g, real_topk)
                reluIn = self._safe_mm(Q_index, K_selected.T)  # (g, real_topk)

                # Step 3: ReLU backward - mask where reluIn <= 0
                relu_mask = reluIn > 0
                reluGrad = mulGrad * relu_mask.float()  # (g, real_topk)

                # Step 4: dQ = reluGrad @ K_selected -> (g, real_topk) @ (real_topk, D) -> (g, D)
                dQ_index = self._safe_mm(reluGrad, K_selected)  # (g, D)
                dq_batch[s1_idx, :, :] = dQ_index.reshape(N1, D)

                # Step 5: dK = scatter_add(reluGrad^T @ Q) -> (real_topk, g) @ (g, D) -> (real_topk, D)
                reluGrad_T = reluGrad.T  # (real_topk, g)
                dK_partial = self._safe_mm(reluGrad_T, Q_index)  # (real_topk, D)
                # Scatter add to dk_batch
                dk_batch.scatter_add_(0, valid_indices.unsqueeze(-1).expand(-1, D), dK_partial)

                # Step 6: dW = sum(ReLU(reluIn) * dV, dim=-1) -> (g,)
                reluOut = torch.clamp(reluIn, min=0.0)  # (g, real_topk)
                dW_index = torch.sum(reluOut * dv_valid.unsqueeze(0), dim=-1)  # (g,)
                dw_batch[s1_idx, :] = dW_index.reshape(N1)

            # Store batch gradients back
            if layout_query == 'BSND':
                dq[batch_idx, :curr_s1, :, :] = dq_batch
                dw[batch_idx, :curr_s1, :] = dw_batch
            else:  # TND
                dq[q_offset:q_offset + curr_s1, :, :] = dq_batch
                dw[q_offset:q_offset + curr_s1, :] = dw_batch

            # Handle key gradient based on layout
            if layout_key == 'PA_BSND':
                # For PA_BSND, need to scatter back to block format
                # Get block_size from key shape
                _, blk_size, _, _ = key.shape
                # Only iterate over the blocks actually used (same as _get_data_from_pa_cache)
                need_block_num = (curr_s2 + blk_size - 1) // blk_size
                for block_idx in range(need_block_num):
                    block_id = block_table[batch_idx][block_idx].item()
                    start = block_idx * blk_size
                    end = min(start + blk_size, curr_s2)
                    dk[block_id, :end - start, 0, :] += dk_batch[start:end, :]
            elif layout_key == 'BSND':
                dk[batch_idx, :curr_s2, 0, :] += dk_batch
            else:  # TND
                dk[k_offset:k_offset + curr_s2, 0, :] += dk_batch

            # Update offsets
            q_offset += curr_s1
            k_offset += curr_s2

        # Convert to input dtype
        input_dtype = query.dtype
        dq = dq.to(input_dtype)
        dk = dk.to(input_dtype)
        dw = dw.to(input_dtype)

        return dq, dk, dw

def torch_cpu_lightning_indexer_forward_backward(query, key, weights, actual_seq_lengths_query=None,
                                                 actual_seq_lengths_key=None, block_table=None, layout_query='BSND',
                                                 layout_key='BSND', sparse_count=2048, sparse_mode=3,
                                                 pre_tokens=9223372036854775807, next_tokens=9223372036854775807,
                                                 return_value=False, grad=False):
    cpu_gloden_exec = TorchCpuLightningIndexer(query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key,
                                               block_table, layout_query, layout_key, sparse_count, sparse_mode,
                                               return_value)
    sparse_indices, sparse_values = cpu_gloden_exec.forward()
    if grad:
        # Create value_grad as ones (similar to backward(gradient=torch.ones_like(sparse_values)))
        value_grad = torch.ones_like(sparse_indices, dtype=query.dtype)
        dq, dk, dw = cpu_gloden_exec.backward(sparse_indices, value_grad)
        return sparse_indices, sparse_values, dq, dk, dw
    return sparse_indices, sparse_values

def torch_gloden_expect_forward_backward(query, key, weights, actual_seq_lengths_query=None,
                                         actual_seq_lengths_key=None, block_table=None, layout_query='BSND',
                                         layout_key='BSND', sparse_count=2048, sparse_mode=3,
                                         pre_tokens=9223372036854775807, next_tokens=9223372036854775807,
                                         return_value=False, grad=False):
    if _env_has_torch_npu:
        gloden_func = torch_npu_lightning_indexer_forward_backward
    else:
        gloden_func = torch_cpu_lightning_indexer_forward_backward

    return gloden_func(query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key, block_table, layout_query,
                       layout_key, sparse_count, sparse_mode, pre_tokens, next_tokens, return_value, grad)

def set_context_mode(mode):
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'kbk':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O2')

def get_ndarrays(layout_query, layout_key, *, B=None, S1=None, S2=None, N1=None, N2=None, D=None, T=None,
                 block_size=None, dtype='fp16'):
    def _generate_actual_seq_lengths_TND(B, T):
        if B < 1:
            raise ValueError("B must be at least 1")
        if B > T:
            raise ValueError(f"Cannot generate {B} unique values ending at {T}")
        if B == 1:
            return np.array([T], dtype=np.int32)
        step = (T - 1) / (B - 1)
        if step < 1:
            raise ValueError(f"Cannot generate {B} unique values from 1 to {T} with constant step >= 1")
        step = int(step)
        start = T - (B - 1) * step
        if start < 1:
            start = 1
            step = (T - start) // (B - 1)
            step = max(step, 1)
        sequence = np.array([start + i * step for i in range(B)], dtype=np.int32)
        sequence[-1] = T
        return sequence

    # Set random seed for reproducibility when torch_npu is not available
    if not _env_has_torch_npu:
        if dtype == 'fp16':
            np.random.seed(2101434266)
        else:
            np.random.seed(833981265)

    # Initialize variables to avoid possibly-used-before-assignment errors
    query = None
    key = None
    weights = None
    actual_seq_lengths_query = None
    actual_seq_lengths_key = None
    block_table = None

    # query weights actual_seq_lengths_query
    if layout_query == 'BSND':
        query = np.random.uniform(-10, 10, (B, S1, N1, D)).astype(np.float32)
        weights = np.random.uniform(-1, 1, (B, S1, N1)).astype(np.float32)
        actual_seq_lengths_query = np.random.uniform(S1, S1, (B,)).astype(np.int32)
    if layout_query == 'TND':
        query = np.random.uniform(-10, 10, (T, N1, D)).astype(np.float32)
        weights = np.random.uniform(-1, 1, (T, N1)).astype(np.float32)
        actual_seq_lengths_query = _generate_actual_seq_lengths_TND(B, T)
    # key actual_seq_lengths_key block_table
    if layout_key == 'PA_BSND':
        key = np.random.uniform(-10, 10, (B * (S2 // block_size), block_size, N2, D)).astype(np.float32)
        block_table = np.array([range(B * S2 // block_size)]).astype(np.int32).reshape((B, -1))
        actual_seq_lengths_key = np.random.uniform(S2, S2, (B,)).astype(np.int32)
    if layout_key == 'BSND':
        key = np.random.uniform(-10, 10, (B, S2, N2, D)).astype(np.float32)
        actual_seq_lengths_key = np.random.uniform(S2, S2, (B,)).astype(np.int32)
        block_table = None
    if layout_key == 'TND':
        key = np.random.uniform(-10, 10, (T, N2, D)).astype(np.float32)
        actual_seq_lengths_key = _generate_actual_seq_lengths_TND(B, T)
        block_table = None

    return query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key, block_table

def get_mindspore_tensors(query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key, block_table,
                          dtype):
    if dtype == 'fp16':
        query = ms.Tensor(query_np, dtype=ms.float16) if query_np is not None else None
        key = ms.Tensor(key_np, dtype=ms.float16) if key_np is not None else None
        weights = ms.Tensor(weights_np, dtype=ms.float16) if weights_np is not None else None
    else:
        query = ms.Tensor(query_np, dtype=ms.bfloat16) if query_np is not None else None
        key = ms.Tensor(key_np, dtype=ms.bfloat16) if key_np is not None else None
        weights = ms.Tensor(weights_np, dtype=ms.bfloat16) if weights_np is not None else None
    actual_seq_lengths_query = ms.Tensor(actual_seq_lengths_query) if actual_seq_lengths_query is not None else None
    actual_seq_lengths_key = ms.Tensor(actual_seq_lengths_key) if actual_seq_lengths_key is not None else None
    block_table = ms.Tensor(block_table) if block_table is not None else None

    return query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key, block_table

def get_torch_tensors(query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key, block_table,
                      dtype):
    if dtype == 'fp16':
        query = torch.tensor(query_np, dtype=torch.float16) if query_np is not None else None
        key = torch.tensor(key_np, dtype=torch.float16) if key_np is not None else None
        weights = torch.tensor(weights_np, dtype=torch.float16) if weights_np is not None else None
    else:
        query = torch.tensor(query_np, dtype=torch.bfloat16) if query_np is not None else None
        key = torch.tensor(key_np, dtype=torch.bfloat16) if key_np is not None else None
        weights = torch.tensor(weights_np, dtype=torch.bfloat16) if weights_np is not None else None
    actual_seq_lengths_query = torch.tensor(actual_seq_lengths_query) if actual_seq_lengths_query is not None else None
    actual_seq_lengths_key = torch.tensor(actual_seq_lengths_key) if actual_seq_lengths_key is not None else None
    block_table = torch.tensor(block_table) if block_table is not None else None

    return query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key, block_table

def compare_results(torch_out, ms_out, sparse_count, return_value=False, skip_cpu_compare=True,
                    skip_cpu_grad_compare=True):
    """
    Compare results between torch and mindspore outputs.

    Args:
        torch_out: Tuple of torch outputs (sparse_indices, sparse_values, dq, dk, dw)
        ms_out: Tuple of mindspore outputs
        sparse_count: The sparse count value
        return_value: Whether sparse_values is returned
        skip_cpu_compare: If True, only compare shape/dtype for forward outputs (sparse_indices, sparse_values)
                          in CPU mode; if False, compare actual values
        skip_cpu_grad_compare: If True, only compare shape/dtype for backward outputs (dq, dk, dw)
                               in CPU mode; if False, compare actual values
    """
    def get_ndarray_md5(arr):
        arr_bytes = np.ascontiguousarray(arr).tobytes()
        return hashlib.md5(arr_bytes).hexdigest()

    def compare_with_torch(torch_out, ms_out):
        _indices_to_name = {
            0: 'sparse_indices',
            1: 'sparse_values',
            2: 'dq',
            3: 'dk',
            4: 'dw',
        }
        # Forward outputs are indices 0, 1; backward outputs are indices 2, 3, 4
        _forward_name = {'sparse_indices', 'sparse_values'}
        _backward_name = {'dq', 'dk', 'dw'}

        for idx, (expect, actual) in enumerate(zip(torch_out, ms_out)):
            print(f"comparing mindspore with torch {'npu' if _env_has_torch_npu else 'cpu'} "
                  f"'{_indices_to_name[idx]}'...")
            if _env_has_torch_npu:
                actual_np = actual.float().asnumpy() if actual.dtype == ms.bfloat16 else actual.asnumpy()
                expect_np = expect.float().cpu().numpy() if expect.dtype == torch.bfloat16 else expect.cpu().numpy()
                if _indices_to_name[idx] == 'sparse_values' and not return_value:
                    print(f"skip comparing '{_indices_to_name[idx]}' because return_value is {return_value}.")
                elif _indices_to_name[idx] == 'dk':
                    print(f"skip comparing for '{_indices_to_name[idx]}', "
                          f"because '{_indices_to_name[idx]}' does not supported deterministic yet.")
                else:
                    assert get_ndarray_md5(expect_np) == get_ndarray_md5(actual_np)
            else:
                # Determine if this is a forward or backward output
                is_forward = _indices_to_name[idx] in _forward_name
                is_backward = _indices_to_name[idx] in _backward_name

                # Decide whether to skip value comparison based on output type
                skip_value_compare = (is_forward and skip_cpu_compare) or (is_backward and skip_cpu_grad_compare)

                if skip_value_compare:
                    if actual is not None and expect is not None:
                        actual_np = actual.float().asnumpy() if actual.dtype == ms.bfloat16 else actual.asnumpy()
                        expect_np = expect.float().numpy() if expect.dtype == torch.bfloat16 else expect.numpy()
                        assert actual_np.shape == expect_np.shape, f"{_indices_to_name[idx]} shape mismatch"
                        assert actual_np.dtype == expect_np.dtype, f"{_indices_to_name[idx]} dtype mismatch"
                    else:
                        print(f"skip comparing '{_indices_to_name[idx]}' because one or both are None.")
                else:
                    # CPU golden comparison with actual value check
                    if expect is None or actual is None:
                        if expect is None and actual is None:
                            print(f"skip comparing '{_indices_to_name[idx]}' because both are None.")
                        else:
                            print(f"skip comparing '{_indices_to_name[idx]}' because one of them is None.")
                        continue
                    actual_np = actual.float().asnumpy() if actual.dtype == ms.bfloat16 else actual.asnumpy()
                    expect_np = expect.float().numpy() if expect.dtype == torch.bfloat16 else expect.numpy()
                    if _indices_to_name[idx] == 'sparse_indices':
                        actual_np = actual_np.reshape(-1, sparse_count)
                        expect_np = expect_np.reshape(-1, sparse_count)
                        np.testing.assert_array_equal(expect_np, actual_np)
                    elif _indices_to_name[idx] == 'sparse_values':
                        if not return_value:
                            print(f"skip comparing '{_indices_to_name[idx]}' because return_value is {return_value}.")
                        else:
                            np.testing.assert_allclose(expect_np, actual_np, rtol=1e-2, atol=1e-3,
                                                       err_msg=f"{_indices_to_name[idx]} value mismatch")
                    elif _indices_to_name[idx] in ['dq', 'dk', 'dw']:
                        # For gradient outputs, check shape match and allow for numerical tolerance
                        assert actual_np.shape == expect_np.shape, f"{_indices_to_name[idx]} shape mismatch"
                        # Use relative tolerance for gradient comparison
                        np.testing.assert_allclose(expect_np, actual_np, rtol=1e-2, atol=1e-3,
                                                   err_msg=f"{_indices_to_name[idx]} value mismatch")
                    else:
                        raise ValueError(f"Unknown index {idx}")

    # compare mindspore with torch
    compare_with_torch(torch_out, ms_out)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
def test_ops_lightning_indexer_bsnd_pabsnd_normal(mode, dtype):
    """
    Feature: pyboost function.
    Description: Test function lightning_indexer forward and backward computation.
        Test scenario: BSND layout with PA_BSND key layout, B=1, S1=1, S2=8192, N1=64, N2=1, D=128,
        block_size=256, sparse_count=2048, sparse_mode=3 (rightDownCausal).
    Expectation: Expect correct sparse_indices output and gradient results matching torch_npu reference.
    """
    set_context_mode(mode)

    B, S1, S2, N1, N2, D, T = 1, 1, 8192, 64, 1, 128, 8192
    block_size = 256
    layout_query = 'BSND'
    layout_key = 'PA_BSND'
    sparse_count = 2048
    sparse_mode = 3
    return_value = False

    query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key, block_table = \
        get_ndarrays(
            layout_query, layout_key, B=B, S1=S1, S2=S2, N1=N1, N2=N2, D=D, T=T, block_size=block_size, dtype=dtype
        )
    ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query, ms_actual_seq_lengths_key, ms_block_table = \
        get_mindspore_tensors(
            query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key, block_table, dtype
        )
    pt_query, pt_key, pt_weights, pt_actual_seq_lengths_query, pt_actual_seq_lengths_key, pt_block_table = \
        get_torch_tensors(
            query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key, block_table, dtype
        )
    # torch lightning_indexer forward and backward
    torch_out = torch_gloden_expect_forward_backward(pt_query, pt_key, pt_weights, pt_actual_seq_lengths_query,
                                                     pt_actual_seq_lengths_key, pt_block_table, layout_query,
                                                     layout_key, sparse_count, sparse_mode, return_value=return_value,
                                                     grad=True)
    # mindspore lightning_indexer forward
    ms_out = lightning_indexer_forward_func(ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query,
                                            ms_actual_seq_lengths_key, ms_block_table, layout_query, layout_key,
                                            sparse_count, sparse_mode, return_value=return_value)
    # mindspore lightning_indexer backward
    ms_out_grad = lightning_indexer_backward_func(ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query,
                                                  ms_actual_seq_lengths_key, ms_block_table, layout_query, layout_key,
                                                  sparse_count, sparse_mode, return_value=return_value)
    compare_results(torch_out, ms_out + ms_out_grad, sparse_count, return_value, skip_cpu_compare=False)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
def test_ops_lightning_indexer_tnd_pabsnd_forward(mode, dtype):
    """
    Feature: pyboost function.
    Description: Test function lightning_indexer forward computation with TND layout.
        Test scenario: TND layout with PA_BSND key layout, B=3, T=5, S2=8192, N1=64, N2=1, D=128,
        block_size=256, sparse_count=2048, sparse_mode=3 (rightDownCausal).
    Expectation: Expect correct sparse_indices output matching torch_npu reference.
    """
    set_context_mode(mode)

    B, T, S2, N1, N2, D, = 3, 5, 8192, 64, 1, 128
    block_size = 256
    layout_query = 'TND'
    layout_key = 'PA_BSND'
    sparse_count = 2048
    sparse_mode = 3
    return_value = False

    query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key, block_table = get_ndarrays(
        layout_query, layout_key, B=B, T=T, S2=S2, N1=N1, N2=N2, D=D, block_size=block_size, dtype=dtype)
    ms_tensors = get_mindspore_tensors(query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key,
                                       block_table, dtype)
    pt_tensors = get_torch_tensors(query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key,
                                   block_table, dtype)
    ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query, ms_actual_seq_lengths_key, ms_block_table = ms_tensors
    pt_query, pt_key, pt_weights, pt_actual_seq_lengths_query, pt_actual_seq_lengths_key, pt_block_table = pt_tensors

    # torch lightning_indexer forward and backward
    torch_out = torch_gloden_expect_forward_backward(pt_query, pt_key, pt_weights, pt_actual_seq_lengths_query,
                                                     pt_actual_seq_lengths_key, pt_block_table, layout_query,
                                                     layout_key, sparse_count, sparse_mode, return_value=return_value)
    # mindspore lightning_indexer forward
    ms_out = lightning_indexer_forward_func(ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query,
                                            ms_actual_seq_lengths_key, ms_block_table, layout_query, layout_key,
                                            sparse_count, sparse_mode, return_value=return_value)
    compare_results(torch_out, ms_out, sparse_count, return_value, skip_cpu_compare=False)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
def test_ops_lightning_indexer_tnd_tnd_normal(mode, dtype):
    """
    Feature: pyboost function.
    Description: Test function lightning_indexer forward and backward computation with TND layout.
        Test scenario: TND layout with TND key layout (non-PA scenario), B=3, T=5, S2=2048, N1=64, N2=1, D=128,
        sparse_count=2048, sparse_mode=3 (rightDownCausal).
    Expectation: Expect correct sparse_indices output and gradient results matching torch_npu reference.
    """
    set_context_mode(mode)

    B, T, S2, N1, N2, D = 3, 5, 2048, 64, 1, 128
    layout_query = 'TND'
    layout_key = 'TND'
    sparse_count = 2048
    sparse_mode = 3
    return_value = False

    query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key, _ = \
        get_ndarrays(layout_query, layout_key, B=B, T=T, S2=S2, N1=N1, N2=N2, D=D, dtype=dtype)
    ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query, ms_actual_seq_lengths_key, _ = \
        get_mindspore_tensors(query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key,
                              None, dtype)
    pt_query, pt_key, pt_weights, pt_actual_seq_lengths_query, pt_actual_seq_lengths_key, _ = \
        get_torch_tensors(query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key,
                          None, dtype)

    # torch lightning_indexer forward and backward
    torch_out = torch_gloden_expect_forward_backward(pt_query, pt_key, pt_weights, pt_actual_seq_lengths_query,
                                                     pt_actual_seq_lengths_key, None, layout_query,
                                                     layout_key, sparse_count, sparse_mode, return_value=return_value,
                                                     grad=True)
    # mindspore lightning_indexer forward
    ms_out = lightning_indexer_forward_func(ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query,
                                            ms_actual_seq_lengths_key, None, layout_query, layout_key,
                                            sparse_count, sparse_mode, return_value=return_value)
    # mindspore lightning_indexer backward
    ms_out_grad = lightning_indexer_backward_func(ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query,
                                                  ms_actual_seq_lengths_key, None, layout_query, layout_key,
                                                  sparse_count, sparse_mode, return_value=return_value)
    compare_results(torch_out, ms_out + ms_out_grad, sparse_count, return_value, skip_cpu_grad_compare=False)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
def test_ops_lightning_indexer_bsnd_bsnd_multi_batch(mode, dtype):
    """
    Feature: pyboost function.
    Description: Test function lightning_indexer forward and backward computation with multi-batch BSND layout.
        Test scenario: BSND layout with BSND key layout (non-PA scenario), B=4, S1=8, S2=2048, N1=64, N2=1, D=128,
        sparse_count=2048, sparse_mode=3 (rightDownCausal).
    Expectation: Expect correct sparse_indices output and gradient results matching torch_npu reference.
    """
    set_context_mode(mode)

    B, S1, S2, N1, N2, D = 4, 8, 2048, 64, 1, 128
    layout_query = 'BSND'
    layout_key = 'BSND'
    sparse_count = 2048
    sparse_mode = 3
    return_value = False

    query_np, key_np, weights_np, _, _, _ = \
        get_ndarrays(layout_query, layout_key, B=B, S1=S1, S2=S2, N1=N1, N2=N2, D=D, dtype=dtype)
    ms_query, ms_key, ms_weights, _, _, _ = \
        get_mindspore_tensors(query_np, key_np, weights_np, None, None, None, dtype)
    pt_query, pt_key, pt_weights, _, _, _ = \
        get_torch_tensors(query_np, key_np, weights_np, None, None, None, dtype)

    # torch lightning_indexer forward and backward
    torch_out = torch_gloden_expect_forward_backward(pt_query, pt_key, pt_weights, None, None, None, layout_query,
                                                     layout_key, sparse_count, sparse_mode, return_value=return_value,
                                                     grad=True)
    # mindspore lightning_indexer forward
    ms_out = lightning_indexer_forward_func(ms_query, ms_key, ms_weights, None, None, None, layout_query, layout_key,
                                            sparse_count, sparse_mode, return_value=return_value)
    # mindspore lightning_indexer backward
    ms_out_grad = lightning_indexer_backward_func(ms_query, ms_key, ms_weights, None, None, None, layout_query,
                                                  layout_key, sparse_count, sparse_mode, return_value=return_value)
    compare_results(torch_out, ms_out + ms_out_grad, sparse_count, return_value, skip_cpu_grad_compare=False)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
def test_ops_lightning_indexer_tnd_tnd_multi_batch(mode, dtype):
    """
    Feature: pyboost function.
    Description: Test function lightning_indexer forward and backward computation with multi-batch TND layout.
        Test scenario: TND layout with TND key layout (non-PA scenario), B=4, T=16, S2=2048, N1=64, N2=1, D=128,
        sparse_count=2048, sparse_mode=3 (rightDownCausal).
    Expectation: Expect correct sparse_indices output and gradient results matching torch_npu reference.
    """
    set_context_mode(mode)

    B, T, S2, N1, N2, D = 4, 16, 2048, 64, 1, 128
    layout_query = 'TND'
    layout_key = 'TND'
    sparse_count = 2048
    sparse_mode = 3
    return_value = False

    query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key, _ = \
        get_ndarrays(layout_query, layout_key, B=B, T=T, S2=S2, N1=N1, N2=N2, D=D, dtype=dtype)
    ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query, ms_actual_seq_lengths_key, _ = \
        get_mindspore_tensors(query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key,
                              None, dtype)
    pt_query, pt_key, pt_weights, pt_actual_seq_lengths_query, pt_actual_seq_lengths_key, _ = \
        get_torch_tensors(query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key,
                          None, dtype)

    # torch lightning_indexer forward and backward
    torch_out = torch_gloden_expect_forward_backward(pt_query, pt_key, pt_weights, pt_actual_seq_lengths_query,
                                                     pt_actual_seq_lengths_key, None, layout_query,
                                                     layout_key, sparse_count, sparse_mode, return_value=return_value,
                                                     grad=True)
    # mindspore lightning_indexer forward
    ms_out = lightning_indexer_forward_func(ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query,
                                            ms_actual_seq_lengths_key, None, layout_query, layout_key,
                                            sparse_count, sparse_mode, return_value=return_value)
    # mindspore lightning_indexer backward
    ms_out_grad = lightning_indexer_backward_func(ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query,
                                                  ms_actual_seq_lengths_key, None, layout_query, layout_key,
                                                  sparse_count, sparse_mode, return_value=return_value)
    compare_results(torch_out, ms_out + ms_out_grad, sparse_count, return_value, skip_cpu_grad_compare=False)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
def test_ops_lightning_indexer_default_params(mode, dtype):
    """
    Feature: pyboost function.
    Description: Test function lightning_indexer forward and backward with default keyword arguments.
        Test scenario: BSND layout with BSND key layout (default), B=1, S1=1, S2=2048, N1=64, N2=1, D=128.
        All keyword arguments use default values: sparse_count=2048, sparse_mode=3, block_table=None.
    Expectation: Expect correct sparse_indices and gradient output matching torch_npu reference.
    """
    set_context_mode(mode)

    B, S1, S2, N1, N2, D = 1, 1, 2048, 64, 1, 128
    layout_query = 'BSND'
    layout_key = 'BSND'
    sparse_count = 2048  # default value for compare_results

    query_np, key_np, weights_np, _, _, _ = \
        get_ndarrays(layout_query, layout_key, B=B, S1=S1, S2=S2, N1=N1, N2=N2, D=D, dtype=dtype)
    ms_query, ms_key, ms_weights, _, _, _ = \
        get_mindspore_tensors(query_np, key_np, weights_np, None, None, None, dtype)
    pt_query, pt_key, pt_weights, _, _, _ = \
        get_torch_tensors(query_np, key_np, weights_np, None, None, None, dtype)

    # torch lightning_indexer forward and backward with default keyword arguments
    torch_out = torch_gloden_expect_forward_backward(pt_query, pt_key, pt_weights, grad=True)
    # mindspore lightning_indexer forward with default keyword arguments
    ms_out = lightning_indexer_forward_func(ms_query, ms_key, ms_weights)
    # mindspore lightning_indexer backward with default keyword arguments
    ms_out_grad = lightning_indexer_backward_func(ms_query, ms_key, ms_weights)
    compare_results(torch_out, ms_out + ms_out_grad, sparse_count, skip_cpu_grad_compare=False)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
def test_ops_lightning_indexer_return_value_true(mode, dtype):
    """
    Feature: pyboost function.
    Description: Test function lightning_indexer forward and backward with return_value=True.
        Test scenario: BSND layout with BSND key layout, B=1, S1=1, S2=2048, N1=64, N2=1, D=128.
        Only return_value=True is configured, other keyword arguments use default values.
    Expectation: Expect correct sparse_indices, sparse_values and gradient output matching torch_npu reference.
    """
    set_context_mode(mode)

    B, S1, S2, N1, N2, D = 1, 1, 2048, 64, 1, 128
    layout_query = 'BSND'
    layout_key = 'BSND'
    sparse_count = 2048  # default value for compare_results
    return_value = True

    query_np, key_np, weights_np, _, _, _ = \
        get_ndarrays(layout_query, layout_key, B=B, S1=S1, S2=S2, N1=N1, N2=N2, D=D, dtype=dtype)
    ms_query, ms_key, ms_weights, _, _, _ = \
        get_mindspore_tensors(query_np, key_np, weights_np, None, None, None, dtype)
    pt_query, pt_key, pt_weights, _, _, _ = \
        get_torch_tensors(query_np, key_np, weights_np, None, None, None, dtype)

    # torch lightning_indexer forward and backward with return_value=True
    torch_out = torch_gloden_expect_forward_backward(pt_query, pt_key, pt_weights, return_value=return_value, grad=True)
    # mindspore lightning_indexer forward with return_value=True
    ms_out = lightning_indexer_forward_func(ms_query, ms_key, ms_weights, return_value=return_value)
    # mindspore lightning_indexer backward with return_value=True
    ms_out_grad = lightning_indexer_backward_func(ms_query, ms_key, ms_weights, return_value=return_value)
    compare_results(torch_out, ms_out + ms_out_grad, sparse_count, return_value, skip_cpu_grad_compare=False)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
def test_ops_lightning_indexer_sparse_mode_0(mode, dtype):
    """
    Feature: pyboost function.
    Description: Test function lightning_indexer forward and backward with sparse_mode=0 (defaultMask mode).
        Test scenario: BSND layout with PA_BSND key layout, B=1, S1=1, S2=8192, N1=64, N2=1, D=128,
        block_size=256, sparse_count=2048, sparse_mode=0 (defaultMask, no causal mask).
    Expectation: Expect correct sparse_indices and gradient output matching torch_npu reference.
    """
    set_context_mode(mode)

    B, S1, S2, N1, N2, D, T = 1, 1, 8192, 64, 1, 128, 8192
    block_size = 256
    layout_query = 'BSND'
    layout_key = 'PA_BSND'
    sparse_count = 2048
    sparse_mode = 0  # defaultMask mode
    return_value = False

    query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key, block_table = \
        get_ndarrays(layout_query, layout_key, B=B, S1=S1, S2=S2, N1=N1, N2=N2, D=D, T=T, block_size=block_size,
                     dtype=dtype)
    ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query, ms_actual_seq_lengths_key, ms_block_table = \
        get_mindspore_tensors(query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key,
                              block_table, dtype)
    pt_query, pt_key, pt_weights, pt_actual_seq_lengths_query, pt_actual_seq_lengths_key, pt_block_table = \
        get_torch_tensors(query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key,
                          block_table, dtype)

    # torch lightning_indexer forward and backward
    torch_out = torch_gloden_expect_forward_backward(pt_query, pt_key, pt_weights, pt_actual_seq_lengths_query,
                                                     pt_actual_seq_lengths_key, pt_block_table, layout_query,
                                                     layout_key, sparse_count, sparse_mode, return_value=return_value,
                                                     grad=True)
    # mindspore lightning_indexer forward
    ms_out = lightning_indexer_forward_func(ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query,
                                            ms_actual_seq_lengths_key, ms_block_table, layout_query, layout_key,
                                            sparse_count, sparse_mode, return_value=return_value)
    # mindspore lightning_indexer backward
    ms_out_grad = lightning_indexer_backward_func(ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query,
                                                  ms_actual_seq_lengths_key, ms_block_table, layout_query, layout_key,
                                                  sparse_count, sparse_mode, return_value=return_value)
    compare_results(torch_out, ms_out + ms_out_grad, sparse_count, return_value, skip_cpu_compare=False)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
@pytest.mark.parametrize('sparse_count', [1, 512, 2048])
def test_ops_lightning_indexer_sparse_count_variants(mode, sparse_count):
    """
    Feature: pyboost function.
    Description: Test function lightning_indexer forward and backward with various sparse_count boundary values.
        Test scenario: BSND layout with PA_BSND key layout, B=1, S1=1, S2=8192, N1=64, N2=1, D=128,
        block_size=256, sparse_count=[1, 512, 2048] (boundary values within valid range [1, 2048]).
    Expectation: Expect correct sparse_indices and gradient output for all sparse_count values matching torch_npu.
    """
    set_context_mode(mode)

    B, S1, S2, N1, N2, D, T = 1, 1, 8192, 64, 1, 128, 8192
    block_size = 256
    layout_query = 'BSND'
    layout_key = 'PA_BSND'
    sparse_mode = 3
    return_value = False
    dtype = 'fp16'

    query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key, block_table = \
        get_ndarrays(layout_query, layout_key, B=B, S1=S1, S2=S2, N1=N1, N2=N2, D=D, T=T, block_size=block_size,
                     dtype=dtype)
    ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query, ms_actual_seq_lengths_key, ms_block_table = \
        get_mindspore_tensors(query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key,
                              block_table, dtype)
    pt_query, pt_key, pt_weights, pt_actual_seq_lengths_query, pt_actual_seq_lengths_key, pt_block_table = \
        get_torch_tensors(query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key,
                          block_table, dtype)

    # torch lightning_indexer forward and backward
    torch_out = torch_gloden_expect_forward_backward(pt_query, pt_key, pt_weights, pt_actual_seq_lengths_query,
                                                     pt_actual_seq_lengths_key, pt_block_table, layout_query,
                                                     layout_key, sparse_count, sparse_mode, return_value=return_value,
                                                     grad=True)
    # mindspore lightning_indexer forward
    ms_out = lightning_indexer_forward_func(ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query,
                                            ms_actual_seq_lengths_key, ms_block_table, layout_query, layout_key,
                                            sparse_count, sparse_mode, return_value=return_value)
    # mindspore lightning_indexer backward
    ms_out_grad = lightning_indexer_backward_func(ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query,
                                                  ms_actual_seq_lengths_key, ms_block_table, layout_query, layout_key,
                                                  sparse_count, sparse_mode, return_value=return_value)
    compare_results(torch_out, ms_out + ms_out_grad, sparse_count, return_value, skip_cpu_compare=False)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
@pytest.mark.parametrize('dtype', ['fp16'])
def test_ops_lightning_indexer_empty_tensor(mode, dtype):
    """
    Feature: pyboost function.
    Description: Test function lightning_indexer with various empty tensor inputs.

    This test covers empty tensor scenarios to validate the algorithm's robustness:
    - Test case 1: query empty (S1=0) - Should be rejected by infershape
    - Test case 2: key empty (S2=0) - Should be rejected by infershape
    - Test case 3: weights empty (N1=0) while query/key have valid S dimensions

    Expectation: Query/key empty scenarios raise RuntimeError,
    weights empty scenario returns correct empty tensor shapes.
    """
    set_context_mode(mode)

    layout_query = 'BSND'
    layout_key = 'BSND'
    sparse_count = 128
    sparse_mode = 3

    # Test case 1: Empty query sequence (S1=0) - Should fail (rejected by infershape)
    B, S1, S2, N1, N2, D = 1, 0, 256, 64, 1, 128
    query_np = np.array([]).reshape(B, 0, N1, D).astype(np.float32)
    key_np = np.random.uniform(-10, 10, (B, S2, N2, D)).astype(np.float32)
    weights_np = np.array([]).reshape(B, 0, N1).astype(np.float32)
    actual_seq_lengths_query = np.array([0], dtype=np.int32)
    actual_seq_lengths_key = np.array([S2], dtype=np.int32)

    ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query, ms_actual_seq_lengths_key, _ = \
        get_mindspore_tensors(query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key,
                              None, dtype)

    with pytest.raises(ValueError):
        _ = lightning_indexer_forward_func(ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query,
                                           ms_actual_seq_lengths_key, None, layout_query, layout_key,
                                           sparse_count, sparse_mode)
        _pynative_executor.sync()

    # Test case 2: Empty key sequence (S2=0) - Should fail (rejected by infershape)
    B, S1, S2, N1, N2, D = 1, 2, 0, 64, 1, 128
    query_np = np.random.uniform(-10, 10, (B, S1, N1, D)).astype(np.float32)
    key_np = np.array([]).reshape(B, 0, N2, D).astype(np.float32)
    weights_np = np.random.uniform(-1, 1, (B, S1, N1)).astype(np.float32)
    actual_seq_lengths_query = np.array([S1], dtype=np.int32)
    actual_seq_lengths_key = np.array([0], dtype=np.int32)

    ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query, ms_actual_seq_lengths_key, _ = \
        get_mindspore_tensors(query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key,
                              None, dtype)

    with pytest.raises(ValueError):
        _ = lightning_indexer_forward_func(ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query,
                                           ms_actual_seq_lengths_key, None, layout_query, layout_key,
                                           sparse_count, sparse_mode)
        _pynative_executor.sync()

    # Test case 3: Weights empty (shape contains 0) while query/key have valid shapes
    B, S1, S2, N1, N2, D = 1, 2, 256, 64, 1, 128
    query_np = np.random.uniform(-10, 10, (B, S1, N1, D)).astype(np.float32)
    key_np = np.random.uniform(-10, 10, (B, S2, N2, D)).astype(np.float32)
    weights_np = np.array([]).reshape(B, S1, 0).astype(np.float32)  # Empty weights tensor
    actual_seq_lengths_query = np.array([S1], dtype=np.int32)
    actual_seq_lengths_key = np.array([S2], dtype=np.int32)

    ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query, ms_actual_seq_lengths_key, _ = \
        get_mindspore_tensors(query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key,
                              None, dtype)

    with pytest.raises(RuntimeError):
        _ = lightning_indexer_forward_func(ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query,
                                           ms_actual_seq_lengths_key, None, layout_query, layout_key,
                                           sparse_count, sparse_mode)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_ops_lightning_indexer_invalid_constant_scalar_tensors(mode):
    """
    Feature: pyboost function.
    Description: Test function lightning_indexer with scalar constant tensors (shape=[]).
        Test scenario: query, key, weights are all scalar tensors (shape=[]),
        which violate API shape requirements (query should be 4D, key should be 3D/4D, weights should be 3D).
    Expectation: Expect ValueError/RuntimeError/TypeError for invalid scalar tensor inputs.
    """
    set_context_mode(mode)

    # Constant scalar tensors (shape=[])
    ms_query = ms.Tensor(1.0, dtype=ms.float16)
    ms_key = ms.Tensor(1.0, dtype=ms.float16)
    ms_weights = ms.Tensor(1.0, dtype=ms.float16)
    ms_actual_seq_lengths_query = ms.Tensor(1, dtype=ms.int32)
    ms_actual_seq_lengths_key = ms.Tensor(256, dtype=ms.int32)

    layout_query = 'BSND'
    layout_key = 'BSND'
    sparse_count = 128
    sparse_mode = 3

    # Any valid-looking scalar params here still must fail due to invalid input ranks
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        _ = lightning_indexer_forward_func(ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query,
                                           ms_actual_seq_lengths_key, None, layout_query, layout_key,
                                           sparse_count, sparse_mode)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
@pytest.mark.parametrize('dtype', ['bf16'])
def test_ops_lightning_indexer_non_contiguous(mode, dtype):
    """
    Feature: pyboost function.
    Description: Test function lightning_indexer forward and backward with non-contiguous tensors.
        Test scenario: Tensors are made non-contiguous through transpose+permute operations.
        BSND layout with BSND key layout, B=2, S1=4, S2=2048, N1=64, N2=1, D=128.
        All swapped dimensions > 1 to ensure permute creates truly non-contiguous tensors.
    Expectation: Expect correct forward/backward results with non-contiguous tensors matching torch_npu reference.
    """
    set_context_mode(mode)

    # N2 must be 1 per API constraint; swap dimensions that are both > 1
    B, S1, S2, N1, N2, D = 2, 4, 2048, 64, 1, 128
    layout_query = 'BSND'
    layout_key = 'BSND'
    sparse_count = 2048
    sparse_mode = 3
    return_value = False

    query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key, block_table = \
        get_ndarrays(layout_query, layout_key, B=B, S1=S1, S2=S2, N1=N1, N2=N2, D=D, dtype=dtype)

    # Create non-contiguous tensors through transpose operations
    # query: (B, S1, N1, D) -> swap N1 and D (both > 1)
    query_np_t = query_np.transpose(0, 1, 3, 2).copy()  # (B, S1, D, N1)
    # key: (B, S2, N2, D) -> swap B and S2 (both > 1, since N2=1 cannot be swapped effectively)
    key_np_t = key_np.transpose(1, 0, 2, 3).copy()  # (S2, B, N2, D)
    # weights: (B, S1, N1) -> swap S1 and N1 (both > 1)
    weights_np_t = weights_np.transpose(0, 2, 1).copy()  # (B, N1, S1)

    pt_query, pt_key, pt_weights, pt_actual_seq_lengths_query, pt_actual_seq_lengths_key, pt_block_table = \
        get_torch_tensors(query_np_t, key_np_t, weights_np_t, actual_seq_lengths_query, actual_seq_lengths_key,
                          block_table, dtype)
    # Permute back to original shape but non-contiguous
    pt_query_permute = torch.permute(pt_query, (0, 1, 3, 2))  # (B, S1, N1, D)
    pt_key_permute = torch.permute(pt_key, (1, 0, 2, 3))  # (B, S2, N2, D)
    pt_weights_permute = torch.permute(pt_weights, (0, 2, 1))  # (B, S1, N1)

    assert not pt_query_permute.is_contiguous()
    assert not pt_key_permute.is_contiguous()
    assert not pt_weights_permute.is_contiguous()

    ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query, ms_actual_seq_lengths_key, ms_block_table = \
        get_mindspore_tensors(query_np_t, key_np_t, weights_np_t, actual_seq_lengths_query, actual_seq_lengths_key,
                              block_table, dtype)
    # Permute back to original shape but non-contiguous
    ms_query_permute = ops.permute(ms_query, (0, 1, 3, 2))  # (B, S1, N1, D)
    ms_key_permute = ops.permute(ms_key, (1, 0, 2, 3))  # (B, S2, N2, D)
    ms_weights_permute = ops.permute(ms_weights, (0, 2, 1))  # (B, S1, N1)

    assert not ms_query_permute.is_contiguous()
    assert not ms_key_permute.is_contiguous()
    assert not ms_weights_permute.is_contiguous()

    # torch lightning_indexer forward and backward with non-contiguous tensors
    torch_out = torch_gloden_expect_forward_backward(pt_query_permute, pt_key_permute, pt_weights_permute,
                                                     pt_actual_seq_lengths_query, pt_actual_seq_lengths_key,
                                                     pt_block_table, layout_query, layout_key, sparse_count,
                                                     sparse_mode, return_value=return_value, grad=True)
    # mindspore lightning_indexer forward with non-contiguous tensors
    ms_out = lightning_indexer_forward_func(ms_query_permute, ms_key_permute, ms_weights_permute,
                                            ms_actual_seq_lengths_query, ms_actual_seq_lengths_key, ms_block_table,
                                            layout_query, layout_key, sparse_count, sparse_mode,
                                            return_value=return_value)
    # mindspore lightning_indexer backward with non-contiguous tensors
    ms_out_grad = lightning_indexer_backward_func(ms_query_permute, ms_key_permute, ms_weights_permute,
                                                  ms_actual_seq_lengths_query, ms_actual_seq_lengths_key,
                                                  ms_block_table, layout_query, layout_key, sparse_count,
                                                  sparse_mode, return_value=return_value)
    compare_results(torch_out, ms_out + ms_out_grad, sparse_count, return_value, skip_cpu_grad_compare=False)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
@pytest.mark.parametrize('dtype', [ms.float32, ms.float64, ms.int32, ms.int64, ms.uint8, ms.bool_])
def test_ops_lightning_indexer_invalid_dtype(mode, dtype):
    """
    Feature: pyboost function.
    Description: Test function lightning_indexer with invalid/unsupported data types.
        Test scenario: Input tensors with dtype=[float32, float64, int32, int64, uint8, bool_].
        API specification only supports float16 and bfloat16.
    Expectation: Expect TypeError or RuntimeError for unsupported data types.
    """
    set_context_mode(mode)

    B, S1, S2, N1, N2, D = 1, 1, 256, 64, 1, 128
    block_size = 256
    layout_query = 'BSND'
    layout_key = 'PA_BSND'
    sparse_count = 128
    sparse_mode = 3

    query_np = np.random.uniform(-10, 10, (B, S1, N1, D)).astype(np.float32)
    key_np = np.random.uniform(-10, 10, (B * (S2 // block_size), block_size, N2, D)).astype(np.float32)
    weights_np = np.random.uniform(-1, 1, (B, S1, N1)).astype(np.float32)
    actual_seq_lengths_query = np.array([S1], dtype=np.int32)
    actual_seq_lengths_key = np.array([S2], dtype=np.int32)
    block_table = np.array([range(B * S2 // block_size)], dtype=np.int32).reshape((B, -1))

    ms_query = ms.Tensor(query_np, dtype=dtype)
    ms_key = ms.Tensor(key_np, dtype=dtype)
    ms_weights = ms.Tensor(weights_np, dtype=dtype)
    ms_actual_seq_lengths_query = ms.Tensor(actual_seq_lengths_query)
    ms_actual_seq_lengths_key = ms.Tensor(actual_seq_lengths_key)
    ms_block_table = ms.Tensor(block_table)

    with pytest.raises((TypeError, RuntimeError)):
        _ = lightning_indexer_forward_func(ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query,
                                           ms_actual_seq_lengths_key, ms_block_table, layout_query, layout_key,
                                           sparse_count, sparse_mode)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_ops_lightning_indexer_mixed_dtype(mode):
    """
    Feature: pyboost function.
    Description: Test function lightning_indexer with inconsistent data types across input tensors.
        Test scenario: query(float16) + key(bfloat16) + weights(float16), mixed dtypes.
        API specification requires query, key, weights to have the same dtype.
    Expectation: Expect TypeError or RuntimeError for inconsistent data types.
    """
    set_context_mode(mode)

    B, S1, S2, N1, N2, D = 1, 1, 256, 64, 1, 128
    block_size = 256
    layout_query = 'BSND'
    layout_key = 'PA_BSND'
    sparse_count = 128
    sparse_mode = 3

    query_np = np.random.uniform(-10, 10, (B, S1, N1, D)).astype(np.float32)
    key_np = np.random.uniform(-10, 10, (B * (S2 // block_size), block_size, N2, D)).astype(np.float32)
    weights_np = np.random.uniform(-1, 1, (B, S1, N1)).astype(np.float32)
    actual_seq_lengths_query = np.array([S1], dtype=np.int32)
    actual_seq_lengths_key = np.array([S2], dtype=np.int32)
    block_table = np.array([range(B * S2 // block_size)], dtype=np.int32).reshape((B, -1))

    # Mixed fp16 and bf16
    ms_query = ms.Tensor(query_np, dtype=ms.float16)
    ms_key = ms.Tensor(key_np, dtype=ms.bfloat16)  # Different dtype
    ms_weights = ms.Tensor(weights_np, dtype=ms.float16)
    ms_actual_seq_lengths_query = ms.Tensor(actual_seq_lengths_query)
    ms_actual_seq_lengths_key = ms.Tensor(actual_seq_lengths_key)
    ms_block_table = ms.Tensor(block_table)

    with pytest.raises((TypeError, RuntimeError)):
        _ = lightning_indexer_forward_func(ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query,
                                           ms_actual_seq_lengths_key, ms_block_table, layout_query, layout_key,
                                           sparse_count, sparse_mode)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
@pytest.mark.parametrize('layout_query, layout_key', [('INVALID', 'PA_BSND'), ('BSND', 'INVALID'), ('ABC', 'XYZ')])
def test_ops_lightning_indexer_invalid_layout(mode, layout_query, layout_key):
    """
    Feature: pyboost function.
    Description: Test function lightning_indexer with invalid layout string parameters.
        Test scenario: layout_query/layout_key = ['INVALID', 'ABC', 'XYZ'] (invalid values).
        Valid layout_query: BSND, TND. Valid layout_key: PA_BSND, BSND, TND.
    Expectation: Expect ValueError or RuntimeError for invalid layout strings.
    """
    set_context_mode(mode)

    B, S1, S2, N1, N2, D = 1, 1, 256, 64, 1, 128
    block_size = 256
    sparse_count = 128
    sparse_mode = 3

    query_np = np.random.uniform(-10, 10, (B, S1, N1, D)).astype(np.float32)
    key_np = np.random.uniform(-10, 10, (B * (S2 // block_size), block_size, N2, D)).astype(np.float32)
    weights_np = np.random.uniform(-1, 1, (B, S1, N1)).astype(np.float32)
    actual_seq_lengths_query = np.array([S1], dtype=np.int32)
    actual_seq_lengths_key = np.array([S2], dtype=np.int32)
    block_table = np.array([range(B * S2 // block_size)], dtype=np.int32).reshape((B, -1))

    ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query, ms_actual_seq_lengths_key, ms_block_table = \
        get_mindspore_tensors(query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key,
                              block_table, 'fp16')

    with pytest.raises((ValueError, RuntimeError)):
        _ = lightning_indexer_forward_func(ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query,
                                           ms_actual_seq_lengths_key, ms_block_table, layout_query, layout_key,
                                           sparse_count, sparse_mode)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
@pytest.mark.parametrize('sparse_count', [-1, 0, 2049, 10000])
def test_ops_lightning_indexer_invalid_sparse_count(mode, sparse_count):
    """
    Feature: pyboost function.
    Description: Test function lightning_indexer with sparse_count values outside valid range.
        Test scenario: sparse_count=[-1, 0, 2049, 10000] (invalid values).
        Valid range is [1, 2048] according to API specification.
    Expectation: Expect ValueError or RuntimeError for sparse_count outside valid range.
    """
    set_context_mode(mode)

    B, S1, S2, N1, N2, D = 1, 1, 256, 64, 1, 128
    block_size = 256
    layout_query = 'BSND'
    layout_key = 'PA_BSND'
    sparse_mode = 3

    query_np = np.random.uniform(-10, 10, (B, S1, N1, D)).astype(np.float32)
    key_np = np.random.uniform(-10, 10, (B * (S2 // block_size), block_size, N2, D)).astype(np.float32)
    weights_np = np.random.uniform(-1, 1, (B, S1, N1)).astype(np.float32)
    actual_seq_lengths_query = np.array([S1], dtype=np.int32)
    actual_seq_lengths_key = np.array([S2], dtype=np.int32)
    block_table = np.array([range(B * S2 // block_size)], dtype=np.int32).reshape((B, -1))

    ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query, ms_actual_seq_lengths_key, ms_block_table = \
        get_mindspore_tensors(query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key,
                              block_table, 'fp16')

    with pytest.raises((ValueError, RuntimeError)):
        _ = lightning_indexer_forward_func(ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query,
                                           ms_actual_seq_lengths_key, ms_block_table, layout_query, layout_key,
                                           sparse_count, sparse_mode)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
@pytest.mark.parametrize('sparse_mode', [-1, 1, 2, 4, 100])
def test_ops_lightning_indexer_invalid_sparse_mode(mode, sparse_mode):
    """
    Feature: pyboost function.
    Description: Test function lightning_indexer with invalid sparse_mode values.
        Test scenario: sparse_mode=[-1, 1, 2, 4, 100] (invalid values).
        Valid values are 0 (defaultMask) and 3 (rightDownCausal) according to API specification.
    Expectation: Expect ValueError or RuntimeError for invalid sparse_mode values.
    """
    set_context_mode(mode)

    B, S1, S2, N1, N2, D = 1, 1, 256, 64, 1, 128
    block_size = 256
    layout_query = 'BSND'
    layout_key = 'PA_BSND'
    sparse_count = 128

    query_np = np.random.uniform(-10, 10, (B, S1, N1, D)).astype(np.float32)
    key_np = np.random.uniform(-10, 10, (B * (S2 // block_size), block_size, N2, D)).astype(np.float32)
    weights_np = np.random.uniform(-1, 1, (B, S1, N1)).astype(np.float32)
    actual_seq_lengths_query = np.array([S1], dtype=np.int32)
    actual_seq_lengths_key = np.array([S2], dtype=np.int32)
    block_table = np.array([range(B * S2 // block_size)], dtype=np.int32).reshape((B, -1))

    ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query, ms_actual_seq_lengths_key, ms_block_table = \
        get_mindspore_tensors(query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key,
                              block_table, 'fp16')

    with pytest.raises((ValueError, RuntimeError)):
        _ = lightning_indexer_forward_func(ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query,
                                           ms_actual_seq_lengths_key, ms_block_table, layout_query, layout_key,
                                           sparse_count, sparse_mode)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_ops_lightning_indexer_tnd_layout_n1_32(mode):
    """
    Feature: pyboost function.
    Description: Test function lightning_indexer with TND layout and N1=32.
        Test scenario: TND layout with N1=32 (different from doc example N1=64).
        Verify forward and backward accuracy against torch_npu.
    Expectation: Forward and backward results match torch_npu.
    """
    set_context_mode(mode)

    # TND layout with N1=32
    B, T, S2, N1, N2, D = 3, 5, 8192, 32, 1, 128
    block_size = 256
    layout_query = 'TND'
    layout_key = 'PA_BSND'
    sparse_count = 128
    sparse_mode = 3
    dtype = 'fp16'

    query_np = np.random.uniform(-10, 10, (T, N1, D)).astype(np.float32)
    key_np = np.random.uniform(-10, 10, (B * (S2 // block_size), block_size, N2, D)).astype(np.float32)
    weights_np = np.random.uniform(-1, 1, (T, N1)).astype(np.float32)
    # TND layout requires actual_seq_lengths_query as prefix sum
    actual_seq_lengths_query = np.array([1, 3, 5], dtype=np.int32)  # B=3, prefix sum
    actual_seq_lengths_key = np.array([S2, S2, S2], dtype=np.int32)
    block_table = np.array([range(S2 // block_size) for _ in range(B)], dtype=np.int32).reshape((B, -1))

    ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query, ms_actual_seq_lengths_key, ms_block_table = \
        get_mindspore_tensors(query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key,
                              block_table, dtype)
    pt_query, pt_key, pt_weights, pt_actual_seq_lengths_query, pt_actual_seq_lengths_key, pt_block_table = \
        get_torch_tensors(query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key,
                          block_table, dtype)

    # torch lightning_indexer forward
    torch_out = torch_gloden_expect_forward_backward(pt_query, pt_key, pt_weights, pt_actual_seq_lengths_query,
                                                     pt_actual_seq_lengths_key, pt_block_table, layout_query,
                                                     layout_key, sparse_count, sparse_mode)
    # mindspore lightning_indexer forward
    ms_out = lightning_indexer_forward_func(ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query,
                                            ms_actual_seq_lengths_key, ms_block_table, layout_query, layout_key,
                                            sparse_count, sparse_mode)
    compare_results(torch_out, ms_out, sparse_count, return_value=False, skip_cpu_compare=False,
                    skip_cpu_grad_compare=False)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_ops_lightning_indexer_invalid_n2(mode):
    """
    Feature: pyboost function.
    Description: Test function lightning_indexer with invalid N2 head dimension.
        Test scenario: N2=4 (invalid, N2 only supports 1).
        API specification requires N2=1.
    Expectation: Expect ValueError or RuntimeError for invalid N2 value.
    """
    set_context_mode(mode)

    # N2 != 1 (invalid for all layouts)
    # According to doc: "key中的N支持1" and "N2仅支持1"
    B, S1, S2, N1, N2, D = 1, 1, 256, 64, 4, 128  # N2=4 is invalid
    block_size = 256
    layout_query = 'BSND'
    layout_key = 'PA_BSND'
    sparse_count = 128
    sparse_mode = 3

    key_np = np.random.uniform(-10, 10, (B * (S2 // block_size), block_size, N2, D)).astype(np.float32)
    query_np = np.random.uniform(-10, 10, (B, S1, N1, D)).astype(np.float32)
    weights_np = np.random.uniform(-1, 1, (B, S1, N1)).astype(np.float32)
    actual_seq_lengths_query = np.array([S1], dtype=np.int32)
    actual_seq_lengths_key = np.array([S2], dtype=np.int32)
    block_table = np.array([range(B * S2 // block_size)], dtype=np.int32).reshape((B, -1))

    ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query, ms_actual_seq_lengths_key, ms_block_table = \
        get_mindspore_tensors(query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key,
                              block_table, 'fp16')

    with pytest.raises((ValueError, RuntimeError)):
        _ = lightning_indexer_forward_func(ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query,
                                           ms_actual_seq_lengths_key, ms_block_table, layout_query, layout_key,
                                           sparse_count, sparse_mode)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_ops_lightning_indexer_invalid_d(mode):
    """
    Feature: pyboost function.
    Description: Test function lightning_indexer with invalid D (head dimension) value.
        Test scenario: D=64 (invalid, should be 128).
        API specification requires query and key D dimension to be 128.
    Expectation: Expect ValueError or RuntimeError for D != 128.
    """
    set_context_mode(mode)

    B, S1, S2, N1, N2, D = 1, 1, 256, 64, 1, 64  # D=64 is invalid
    block_size = 256
    layout_query = 'BSND'
    layout_key = 'PA_BSND'
    sparse_count = 128
    sparse_mode = 3

    query_np = np.random.uniform(-10, 10, (B, S1, N1, D)).astype(np.float32)
    key_np = np.random.uniform(-10, 10, (B * (S2 // block_size), block_size, N2, D)).astype(np.float32)
    weights_np = np.random.uniform(-1, 1, (B, S1, N1)).astype(np.float32)
    actual_seq_lengths_query = np.array([S1], dtype=np.int32)
    actual_seq_lengths_key = np.array([S2], dtype=np.int32)
    block_table = np.array([range(B * S2 // block_size)], dtype=np.int32).reshape((B, -1))

    ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query, ms_actual_seq_lengths_key, ms_block_table = \
        get_mindspore_tensors(query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key,
                              block_table, 'fp16')

    with pytest.raises((ValueError, RuntimeError)):
        _ = lightning_indexer_forward_func(ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query,
                                           ms_actual_seq_lengths_key, ms_block_table, layout_query, layout_key,
                                           sparse_count, sparse_mode)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
@pytest.mark.parametrize('value', [np.inf, -np.inf, np.nan])
def test_ops_lightning_indexer_qkw_inf_nan_values(mode, value):
    """
    Feature: pyboost function.
    Description: Test function lightning_indexer with special floating-point values in input tensors.
        Test scenario: query/key/weights filled with [inf, -inf, nan] special values.
        B=1, S1=1, S2=256, N1=64, N2=1, D=128, sparse_count=128.
    Expectation: Expect correct computation or graceful handling, matching torch_npu behavior.
    """
    if not _env_has_torch_npu:
        print("torch_npu is not installed, skip test because cpu golden does not support special values.")
        return

    set_context_mode(mode)

    B, S1, S2, N1, N2, D = 1, 1, 256, 64, 1, 128
    block_size = 256
    layout_query = 'BSND'
    layout_key = 'PA_BSND'
    sparse_count = 128
    sparse_mode = 3
    dtype = 'fp16'
    return_value = False

    query_np = np.random.uniform(-10, 10, (B, S1, N1, D)).astype(np.float32)
    key_np = np.random.uniform(-10, 10, (B * (S2 // block_size), block_size, N2, D)).astype(np.float32)
    weights_np = np.random.uniform(-1, 1, (B, S1, N1)).astype(np.float32)
    actual_seq_lengths_query = np.array([S1], dtype=np.int32)
    actual_seq_lengths_key = np.array([S2], dtype=np.int32)
    block_table = np.array([range(B * S2 // block_size)], dtype=np.int32).reshape((B, -1))

    # Fill with special values
    query_np.fill(value)
    key_np.fill(value)
    weights_np.fill(value)

    ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query, ms_actual_seq_lengths_key, ms_block_table = \
        get_mindspore_tensors(query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key,
                              block_table, dtype)
    pt_query, pt_key, pt_weights, pt_actual_seq_lengths_query, pt_actual_seq_lengths_key, pt_block_table = \
        get_torch_tensors(query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key,
                          block_table, dtype)

    # torch lightning_indexer forward and backward
    torch_out = torch_gloden_expect_forward_backward(pt_query, pt_key, pt_weights, pt_actual_seq_lengths_query,
                                                     pt_actual_seq_lengths_key, pt_block_table, layout_query,
                                                     layout_key, sparse_count, sparse_mode, return_value=return_value,
                                                     grad=True)
    # mindspore lightning_indexer forward
    ms_out = lightning_indexer_forward_func(ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query,
                                            ms_actual_seq_lengths_key, ms_block_table, layout_query, layout_key,
                                            sparse_count, sparse_mode, return_value=return_value)
    # mindspore lightning_indexer backward
    ms_out_grad = lightning_indexer_backward_func(ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query,
                                                  ms_actual_seq_lengths_key, ms_block_table, layout_query, layout_key,
                                                  sparse_count, sparse_mode, return_value=return_value)
    compare_results(torch_out, ms_out + ms_out_grad, sparse_count, return_value, skip_cpu_compare=False,
                    skip_cpu_grad_compare=False)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_ops_lightning_indexer_invalid_parameter_types(mode):
    """
    Feature: pyboost function.
    Description: Test function lightning_indexer with invalid parameter data types.
        Test scenario: sparse_count="128" (string), sparse_mode=3.0 (float), layout_query=123 (int).
        API requires sparse_count/sparse_mode to be int, layout to be str.
    Expectation: Expect TypeError or ValueError for parameters with invalid data types.
    """
    set_context_mode(mode)

    B, S1, S2, N1, N2, D = 1, 1, 256, 64, 1, 128
    block_size = 256
    layout_query = 'BSND'
    layout_key = 'PA_BSND'

    query_np = np.random.uniform(-10, 10, (B, S1, N1, D)).astype(np.float32)
    key_np = np.random.uniform(-10, 10, (B * (S2 // block_size), block_size, N2, D)).astype(np.float32)
    weights_np = np.random.uniform(-1, 1, (B, S1, N1)).astype(np.float32)
    actual_seq_lengths_query = np.array([S1], dtype=np.int32)
    actual_seq_lengths_key = np.array([S2], dtype=np.int32)
    block_table = np.array([range(B * S2 // block_size)], dtype=np.int32).reshape((B, -1))

    ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query, ms_actual_seq_lengths_key, ms_block_table = \
        get_mindspore_tensors(query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key,
                              block_table, 'fp16')

    # Test invalid sparse_count type (string)
    with pytest.raises((TypeError, ValueError)):
        _ = lightning_indexer_forward_func(ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query,
                                           ms_actual_seq_lengths_key, ms_block_table, layout_query, layout_key,
                                           "128", 3)
        _pynative_executor.sync()

    # Test invalid sparse_mode type (float)
    with pytest.raises((TypeError, ValueError)):
        _ = lightning_indexer_forward_func(ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query,
                                           ms_actual_seq_lengths_key, ms_block_table, layout_query, layout_key,
                                           128, 3.0)
        _pynative_executor.sync()

    # Test invalid layout type (int)
    with pytest.raises((TypeError, ValueError)):
        _ = lightning_indexer_forward_func(ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query,
                                           ms_actual_seq_lengths_key, ms_block_table, 123, layout_key,
                                           128, 3)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
@pytest.mark.parametrize('dtype', ['fp16'])
def test_ops_lightning_indexer_batch_processing(mode, dtype):
    """
    Feature: pyboost function.
    Description: Test function lightning_indexer forward with multiple batch processing using TND layout.
        Test scenario: TND layout with B=4 batches, T=10 total tokens, S2=8192, N1=64, N2=1, D=128.
        actual_seq_lengths_query represents cumulative token counts for each batch.
    Expectation: Expect correct sparse_indices output for multi-batch scenario matching torch_npu.
    """
    set_context_mode(mode)

    B, T, S2, N1, N2, D = 4, 10, 8192, 64, 1, 128
    block_size = 256
    layout_query = 'TND'
    layout_key = 'PA_BSND'
    sparse_count = 2048
    sparse_mode = 3
    return_value = False

    query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key, block_table = get_ndarrays(
        layout_query, layout_key, B=B, T=T, S2=S2, N1=N1, N2=N2, D=D, block_size=block_size, dtype=dtype)
    ms_tensors = get_mindspore_tensors(query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key,
                                       block_table, dtype)
    pt_tensors = get_torch_tensors(query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key,
                                   block_table, dtype)
    ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query, ms_actual_seq_lengths_key, ms_block_table = ms_tensors
    pt_query, pt_key, pt_weights, pt_actual_seq_lengths_query, pt_actual_seq_lengths_key, pt_block_table = pt_tensors

    # torch lightning_indexer forward
    torch_out = torch_gloden_expect_forward_backward(pt_query, pt_key, pt_weights, pt_actual_seq_lengths_query,
                                                     pt_actual_seq_lengths_key, pt_block_table, layout_query,
                                                     layout_key, sparse_count, sparse_mode, return_value=return_value)
    # mindspore lightning_indexer forward
    ms_out = lightning_indexer_forward_func(ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query,
                                            ms_actual_seq_lengths_key, ms_block_table, layout_query, layout_key,
                                            sparse_count, sparse_mode, return_value=return_value)
    compare_results(torch_out, ms_out, sparse_count, return_value, skip_cpu_grad_compare=False)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
@pytest.mark.parametrize('dtype', ['fp16', 'bf16'])
def test_ops_lightning_indexer_bsnd_large_s1_s2(mode, dtype):
    """
    Feature: pyboost function.
    Description: Test function lightning_indexer forward with large S1 and S2 dimensions.
        Test scenario: BSND layout with B=1, S1=64, S2=16384, N1=64, N2=1, D=128.
        Tests forward computation with large query sequence and KV cache size.
    Expectation: Expect correct sparse_indices output matching torch_npu reference.
    """
    set_context_mode(mode)

    B, S1, S2, N1, N2, D = 1, 64, 16384, 64, 1, 128
    block_size = 256
    layout_query = 'BSND'
    layout_key = 'PA_BSND'
    sparse_count = 2048
    sparse_mode = 3
    return_value = False

    query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key, block_table = \
        get_ndarrays(layout_query, layout_key, B=B, S1=S1, S2=S2, N1=N1, N2=N2, D=D, block_size=block_size, dtype=dtype)
    ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query, ms_actual_seq_lengths_key, ms_block_table = \
        get_mindspore_tensors(query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key,
                              block_table, dtype)
    pt_query, pt_key, pt_weights, pt_actual_seq_lengths_query, pt_actual_seq_lengths_key, pt_block_table = \
        get_torch_tensors(query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key,
                          block_table, dtype)

    # torch lightning_indexer forward
    torch_out = torch_gloden_expect_forward_backward(pt_query, pt_key, pt_weights, pt_actual_seq_lengths_query,
                                                     pt_actual_seq_lengths_key, pt_block_table, layout_query,
                                                     layout_key, sparse_count, sparse_mode, return_value=return_value)
    # mindspore lightning_indexer forward
    ms_out = lightning_indexer_forward_func(ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query,
                                            ms_actual_seq_lengths_key, ms_block_table, layout_query, layout_key,
                                            sparse_count, sparse_mode, return_value=return_value)
    compare_results(torch_out, ms_out, sparse_count, return_value, skip_cpu_grad_compare=False)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
@pytest.mark.parametrize('block_size', [256, 512])
def test_ops_lightning_indexer_block_table_boundary(mode, block_size):
    """
    Feature: pyboost function.
    Description: Test function lightning_indexer forward and backward with various block_size configurations.
        Test scenario: block_size=[256, 512] with S2=8192, testing block_table index mapping.
        block_size must be multiple of 16, max supported 1024.
    Expectation: Expect correct sparse_indices and gradient output for different block_size configurations.
    """
    set_context_mode(mode)
    dtype = 'fp16'

    B, S1, S2, N1, N2, D = 1, 1, 8192, 64, 1, 128
    layout_query = 'BSND'
    layout_key = 'PA_BSND'
    sparse_count = 2048
    sparse_mode = 3
    return_value = False

    query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key, block_table = \
        get_ndarrays(layout_query, layout_key, B=B, S1=S1, S2=S2, N1=N1, N2=N2, D=D, block_size=block_size,
                     dtype=dtype)
    ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query, ms_actual_seq_lengths_key, ms_block_table = \
        get_mindspore_tensors(query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key,
                              block_table, dtype)
    pt_query, pt_key, pt_weights, pt_actual_seq_lengths_query, pt_actual_seq_lengths_key, pt_block_table = \
        get_torch_tensors(query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key,
                          block_table, dtype)

    # torch lightning_indexer forward and backward
    torch_out = torch_gloden_expect_forward_backward(pt_query, pt_key, pt_weights, pt_actual_seq_lengths_query,
                                                     pt_actual_seq_lengths_key, pt_block_table, layout_query,
                                                     layout_key, sparse_count, sparse_mode,
                                                     return_value=return_value, grad=True)
    # mindspore lightning_indexer forward
    ms_out = lightning_indexer_forward_func(ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query,
                                            ms_actual_seq_lengths_key, ms_block_table, layout_query, layout_key,
                                            sparse_count, sparse_mode, return_value=return_value)
    # mindspore lightning_indexer backward
    ms_out_grad = lightning_indexer_backward_func(ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query,
                                                  ms_actual_seq_lengths_key, ms_block_table, layout_query,
                                                  layout_key, sparse_count, sparse_mode, return_value=return_value)
    compare_results(torch_out, ms_out + ms_out_grad, sparse_count, return_value, skip_cpu_compare=False)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ops_lightning_indexer_test_op():
    """
    Feature: pyboost function.
    Description: Test function lightning_indexer using TEST_OP framework for comprehensive validation.
        Test scenario: Two input configurations - Input1: B=1, S1=1, S2=8192, sparse_count=2048.
        Input2: B=1, S1=1, S2=256, sparse_count=128. Tests dynamic shape and mode switching.
    Expectation: Expect correct results with TEST_OP framework validation across different configurations.
    """
    def lightning_indexer_func(query, key, weights, actual_seq_lengths_query=None, actual_seq_lengths_key=None,
                               block_table=None, layout_query='BSND', layout_key='BSND', sparse_count=2048,
                               sparse_mode=3, pre_tokens=9223372036854775807, next_tokens=9223372036854775807,
                               return_value=False):
        return ops.lightning_indexer(query, key, weights, actual_seq_lengths_query=actual_seq_lengths_query,
                                     actual_seq_lengths_key=actual_seq_lengths_key, block_table=block_table,
                                     layout_query=layout_query, layout_key=layout_key, sparse_count=sparse_count,
                                     sparse_mode=sparse_mode, pre_tokens=pre_tokens, next_tokens=next_tokens,
                                     return_value=return_value)

    # Input 1: BSND layout
    B1, S1_1, S2_1, N1_1, N2_1, D1 = 1, 1, 8192, 64, 1, 128
    block_size1 = 256
    layout_query1 = 'BSND'
    layout_key1 = 'PA_BSND'
    sparse_count1 = 2048
    sparse_mode1 = 3

    query_np1 = np.random.uniform(-10, 10, (B1, S1_1, N1_1, D1)).astype(np.float32)
    key_np1 = np.random.uniform(-10, 10, (B1 * (S2_1 // block_size1), block_size1, N2_1, D1)).astype(np.float32)
    weights_np1 = np.random.uniform(-1, 1, (B1, S1_1, N1_1)).astype(np.float32)
    actual_seq_lengths_query1 = np.array([S1_1], dtype=np.int32)
    actual_seq_lengths_key1 = np.array([S2_1], dtype=np.int32)
    block_table1 = np.array([range(B1 * S2_1 // block_size1)], dtype=np.int32).reshape((B1, -1))

    ms_query1 = ms.Tensor(query_np1, dtype=ms.bfloat16)
    ms_key1 = ms.Tensor(key_np1, dtype=ms.bfloat16)
    ms_weights1 = ms.Tensor(weights_np1, dtype=ms.bfloat16)
    ms_actual_seq_lengths_query1 = ms.Tensor(actual_seq_lengths_query1)
    ms_actual_seq_lengths_key1 = ms.Tensor(actual_seq_lengths_key1)
    ms_block_table1 = ms.Tensor(block_table1)

    # Input 2: smaller size
    B2, S1_2, S2_2, N1_2, N2_2, D2 = 1, 1, 256, 64, 1, 128
    block_size2 = 256
    layout_query2 = 'BSND'
    layout_key2 = 'PA_BSND'
    sparse_count2 = 128
    sparse_mode2 = 3

    query_np2 = np.random.uniform(-10, 10, (B2, S1_2, N1_2, D2)).astype(np.float32)
    key_np2 = np.random.uniform(-10, 10, (B2 * (S2_2 // block_size2), block_size2, N2_2, D2)).astype(np.float32)
    weights_np2 = np.random.uniform(-1, 1, (B2, S1_2, N1_2)).astype(np.float32)
    actual_seq_lengths_query2 = np.array([S1_2], dtype=np.int32)
    actual_seq_lengths_key2 = np.array([S2_2], dtype=np.int32)
    block_table2 = np.array([range(B2 * S2_2 // block_size2)], dtype=np.int32).reshape((B2, -1))

    ms_query2 = ms.Tensor(query_np2, dtype=ms.bfloat16)
    ms_key2 = ms.Tensor(key_np2, dtype=ms.bfloat16)
    ms_weights2 = ms.Tensor(weights_np2, dtype=ms.bfloat16)
    ms_actual_seq_lengths_query2 = ms.Tensor(actual_seq_lengths_query2)
    ms_actual_seq_lengths_key2 = ms.Tensor(actual_seq_lengths_key2)
    ms_block_table2 = ms.Tensor(block_table2)

    input1 = [ms_query1, ms_key1, ms_weights1, ms_actual_seq_lengths_query1, ms_actual_seq_lengths_key1,
              ms_block_table1, layout_query1, layout_key1, sparse_count1, sparse_mode1]
    input2 = [ms_query2, ms_key2, ms_weights2, ms_actual_seq_lengths_query2, ms_actual_seq_lengths_key2,
              ms_block_table2, layout_query2, layout_key2, sparse_count2, sparse_mode2]

    TEST_OP(
        lightning_indexer_func,
        [input1, input2],
        disable_mode=['GRAPH_MODE_GE'],
        disable_case=['EmptyTensor', 'ScalarTensor', 'Deterministic'],
        case_config={
            'disable_input_check': True,
            'disable_resize': True,
            'ignore_output_index': [1,],
            'ignore_grad_index': [1,]
        },
    )


def lightning_indexer_memory_case(params, actual_seq_qkvlen, dtype, grad_mark):
    """
    Helper function for device memory comparison test.
    """
    def mindspore_forward_func(query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key,
                               block_table, layout_query, layout_key, sparse_count, sparse_mode):
        _, sparse_values = ops.lightning_indexer(
            query, key, weights, actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key, block_table=block_table,
            layout_query=layout_query, layout_key=layout_key, sparse_count=sparse_count,
            sparse_mode=sparse_mode)
        return sparse_values.sum()

    def torch_npu_forward_func(query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key,
                               block_table, layout_query, layout_key, sparse_count, sparse_mode):
        _, sparse_values = torch_npu.npu_lightning_indexer(
            query, key, weights, actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key, block_table=block_table,
            layout_query=layout_query, layout_key=layout_key, sparse_count=sparse_count,
            sparse_mode=sparse_mode)
        return sparse_values.sum()

    if not _env_has_torch_npu:
        print("torch_npu is not installed, skip device memory test")
        return

    B, S1, S2, N1, N2, D = params[0], params[1], params[2], params[3], params[4], params[5]
    block_size = params[6]
    layout_query = 'BSND'
    layout_key = 'PA_BSND'
    sparse_count = 2048
    sparse_mode = 3

    query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key, block_table = \
        get_ndarrays(layout_query, layout_key, B=B, S1=S1, S2=S2, N1=N1, N2=N2, D=D, block_size=block_size, dtype=dtype)

    print(f"\nStart {dtype} grad_position={grad_mark} memory test...\n"
          f"B={B}, S1={S1}, S2={S2}, N1={N1}, N2={N2}, D={D}")

    ms.runtime.reset_max_memory_allocated()
    ms_init_memory = ms.runtime.max_memory_allocated()
    print(f"ms_init_memory = {ms_init_memory}")

    ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query, ms_actual_seq_lengths_key, ms_block_table = \
        get_mindspore_tensors(query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key,
                              block_table, dtype)

    _ = mindspore_forward_func(ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query, ms_actual_seq_lengths_key,
                               ms_block_table, layout_query, layout_key, sparse_count, sparse_mode)
    _pynative_executor.sync()
    ms_forward_memory = ms.runtime.max_memory_allocated()
    print(f"ms_forward_memory = {ms_forward_memory}")

    grad_position = []
    if 'q' in grad_mark:
        grad_position.append(0)
    if 'k' in grad_mark:
        grad_position.append(1)
    if 'w' in grad_mark:
        grad_position.append(2)
    grad_position = tuple(grad_position)

    _ = ms.grad(mindspore_forward_func, grad_position=grad_position)(
        ms_query, ms_key, ms_weights, ms_actual_seq_lengths_query, ms_actual_seq_lengths_key,
        ms_block_table, layout_query, layout_key, sparse_count, sparse_mode)
    _pynative_executor.sync()
    ms_backward_memory = ms.runtime.max_memory_allocated()
    print(f"ms_backward_memory = {ms_backward_memory}")

    ms_forward_memory_diff = ms_forward_memory - ms_init_memory
    ms_backward_memory_diff = ms_backward_memory - ms_init_memory

    torch_npu.npu.reset_max_memory_allocated()
    pt_init_memory = torch_npu.npu.max_memory_allocated()
    print(f"pt_init_memory = {pt_init_memory}")

    pt_query, pt_key, pt_weights, pt_actual_seq_lengths_query, pt_actual_seq_lengths_key, pt_block_table = \
        get_torch_tensors(query_np, key_np, weights_np, actual_seq_lengths_query, actual_seq_lengths_key,
                          block_table, dtype)
    pt_query, pt_key, pt_weights = pt_query.npu(), pt_key.npu(), pt_weights.npu()
    pt_actual_seq_lengths_query = pt_actual_seq_lengths_query.npu()
    pt_actual_seq_lengths_key = pt_actual_seq_lengths_key.npu()
    pt_block_table = pt_block_table.npu()

    if 'q' in grad_mark:
        pt_query.requires_grad = True
    if 'k' in grad_mark:
        pt_key.requires_grad = True
    if 'w' in grad_mark:
        pt_weights.requires_grad = True

    pt_out = torch_npu_forward_func(pt_query, pt_key, pt_weights, pt_actual_seq_lengths_query,
                                    pt_actual_seq_lengths_key, pt_block_table, layout_query, layout_key,
                                    sparse_count, sparse_mode)
    torch.npu.synchronize()
    pt_forward_memory = torch_npu.npu.max_memory_allocated()
    print(f"pt_forward_memory = {pt_forward_memory}")

    pt_out.backward()
    torch.npu.synchronize()
    pt_backward_memory = torch_npu.npu.max_memory_allocated()
    print(f"pt_backward_memory = {pt_backward_memory}")

    pt_forward_memory_diff = pt_forward_memory - pt_init_memory
    pt_backward_memory_diff = pt_backward_memory - pt_init_memory

    print(f"ms_forward_memory_diff = {ms_forward_memory_diff}")
    print(f"ms_backward_memory_diff = {ms_backward_memory_diff}")
    print(f"pt_forward_memory_diff = {pt_forward_memory_diff}")
    print(f"pt_backward_memory_diff = {pt_backward_memory_diff}")

    assert ms_forward_memory_diff <= pt_forward_memory_diff
    assert ms_backward_memory_diff <= pt_backward_memory_diff


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_ops_lightning_indexer_device_memory_grad_q():
    """
    Feature: device memory comparison.
    Description: Compare device memory usage between MindSpore and torch_npu for lightning_indexer.
        Test scenario: B=1, S1=1, S2=8192, N1=64, N2=1, D=128, block_size=256, dtype=fp16, grad_position='q'.
    Expectation: MindSpore max memory increase is less than or equal to torch_npu.
    """
    params = [1, 1, 8192, 64, 1, 128, 256]
    lightning_indexer_memory_case(params, None, 'fp16', 'q')


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_ops_lightning_indexer_device_memory_grad_k():
    """
    Feature: device memory comparison.
    Description: Compare device memory usage between MindSpore and torch_npu for lightning_indexer.
        Test scenario: B=1, S1=1, S2=8192, N1=64, N2=1, D=128, block_size=256, dtype=fp16, grad_position='k'.
    Expectation: MindSpore max memory increase is less than or equal to torch_npu.
    """
    params = [1, 1, 8192, 64, 1, 128, 256]
    lightning_indexer_memory_case(params, None, 'fp16', 'k')


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_ops_lightning_indexer_device_memory_grad_w():
    """
    Feature: device memory comparison.
    Description: Compare device memory usage between MindSpore and torch_npu for lightning_indexer.
        Test scenario: B=1, S1=1, S2=8192, N1=64, N2=1, D=128, block_size=256, dtype=fp16, grad_position='w'.
    Expectation: MindSpore max memory increase is less than or equal to torch_npu.
    """
    params = [1, 1, 8192, 64, 1, 128, 256]
    lightning_indexer_memory_case(params, None, 'fp16', 'w')


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_ops_lightning_indexer_device_memory_grad_qkw():
    """
    Feature: device memory comparison.
    Description: Compare device memory usage between MindSpore and torch_npu for lightning_indexer.
        Test scenario: B=1, S1=1, S2=8192, N1=64, N2=1, D=128, block_size=256, dtype=bf16, grad_position='qkw'.
    Expectation: MindSpore max memory increase is less than or equal to torch_npu.
    """
    params = [1, 1, 8192, 64, 1, 128, 256]
    lightning_indexer_memory_case(params, None, 'bf16', 'qkw')

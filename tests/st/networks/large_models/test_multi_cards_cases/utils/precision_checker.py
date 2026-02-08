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
"""
Precision check tool.
"""
import numpy as np


class PrecisionChecker:
    """
    Check precision between golden data and input data.
    Args:
            cos_sim_thd: cos similarity threshold
            l1_norm_thd: l1 normalization threshold
            kl_dvg_thd: kl divergence threshold
    """
    def __init__(self, cos_sim_thd=0.999, l1_norm_thd=0.01, kl_dvg_thd=0.005):
        self.cos_sim_thd = cos_sim_thd
        self.l1_norm_thd = l1_norm_thd
        self.kl_dvg_thd = kl_dvg_thd

    def check_precision(self, golden_np, input_np):
        """
        Check precision between golden data and input data.
        Args:
            golden_np: golden data, shape (batch_size, seq_length, input_size)
            input_np: input data, shape (batch_size, seq_length, input_size)
        Returns:
            True or False. True means precision check passed, otherwise failed.
        """
        if not golden_np.dtype == input_np.dtype:
            raise ValueError("The dtype of golden data is not the same as input data")
        if not golden_np.shape == input_np.shape:
            raise ValueError("The shape of golden data is not the same as input data")
        golden_flatten = golden_np.flatten()
        input_flatten = input_np.flatten()

        # Compute KL divergence
        kl = self._kl_divergence(golden_flatten, input_flatten)

        # Compute cosine similarity
        cos = self._cosine_similarity_numpy(golden_flatten, input_flatten)

        # Compute l1_norm
        l1_norm = (np.abs(input_flatten).sum() / np.abs(golden_flatten).sum()) - 1

        # Check the precision
        if cos > self.cos_sim_thd and l1_norm < self.l1_norm_thd and kl < self.kl_dvg_thd:
            return True
        raise AssertionError(f"Precision check failed: "
                             f"cos similarity={cos} (required>{self.cos_sim_thd}), "
                             f"l1_norm={l1_norm} (required<{self.l1_norm_thd}), "
                             f"kl={kl} (required<{self.l1_norm_thd}).")

    def _kl_divergence(self, golden_flatten, input_flatten):
        """
        Compute the KL divergence between flattened golden data and flattened input data.
        Args:
            golden_flatten: flattened golden data, shape (batch_size * seq_length * input_size, )
            input_flatten: flattened input data, shape (batch_size * seq_length * input_size, )
        Returns:
            the sum of KL divergence across all elements
        """
        # compute log_softmax
        def log_softmax(x, axis=-1):
            x_max = np.max(x, axis=axis, keepdims=True)
            log_sum_exp = np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))
            return x - x_max - log_sum_exp

        log_golden = log_softmax(golden_flatten, axis=-1)
        log_input = log_softmax(input_flatten, axis=-1)
        # convert from log-prob to prob
        p = np.exp(log_golden)
        kl = p * (log_golden - log_input)
        return np.sum(kl)

    def _cosine_similarity_numpy(self, golden_flatten, input_flatten, axis=-1):
        """
        Compute cosine similarity between flattened golden data and flattened input data.
        Args:
            golden_flatten: flattened golden data, shape (batch_size * seq_length * input_size, )
            input_flatten: flattened input data, shape (batch_size * seq_length * input_size, )
            axis: along this axis to compute similarity, default is -1
        Returns:
            cosine similarity between flattened golden data and flattened input data
        """
        # Compute vector norms
        norm1 = np.linalg.norm(golden_flatten, axis=axis, keepdims=True)
        norm2 = np.linalg.norm(input_flatten, axis=axis, keepdims=True)

        # Compute norm product
        norm_product = norm1 * norm2
        # Avoid division by 0
        zero_mask = norm_product == 0
        norm_product[zero_mask] = 1

        # Compute dot product
        dot_product = np.sum(golden_flatten * input_flatten, axis=axis, keepdims=True)

        # Compute cosine similarity
        cosine_sim = dot_product / norm_product
        # Set cosine similarity of zero vector to 0
        cosine_sim[zero_mask] = 0

        # Remove single dimension
        return np.squeeze(cosine_sim)

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
"""Generate data for mcore GPTModel UT of inference."""
import numpy as np

from mindspore import Tensor

DEFAULT_BATCH_SIZE = 2
DEFAULT_SEQ_LENGTH = 2
DEFAULT_VOCAB_SIZE = 32
DEFAULT_HIDDEN_SIZE = 32
DEFAULT_FFN_HIDDEN_SIZE = 64
DEFAULT_NUM_HEADS = 2
DEFAULT_NUM_BLOCKS = 16
DEFAULT_BLOCK_SIZE = 64


def get_init_params(is_prefill,
                    seq_length=DEFAULT_SEQ_LENGTH,
                    batch_size=DEFAULT_BATCH_SIZE,
                    vocab_size=DEFAULT_VOCAB_SIZE):
    """
    Generates initial parameters (inputs) for the TransformerLayer model.
    """
    np.random.seed(124)
    if is_prefill:
        init_params = {
            "input_ids": Tensor.from_numpy(
                np.random.randint(0, vocab_size, size=(seq_length * batch_size,), dtype=np.int32)),
            "positions": Tensor.from_numpy(
                np.array([i for _ in range(batch_size) for i in range(seq_length)], dtype=np.int32)),
            "attention_mask": Tensor.from_numpy(
                np.triu(np.ones(shape=(128, 128), dtype=np.float16), 1) * -10000.0),
            "batch_valid_length": Tensor.from_numpy(
                np.array([seq_length] * batch_size, dtype=np.int32)),
            "q_seq_lens": Tensor.from_numpy(
                np.array([seq_length] * batch_size, dtype=np.int32)),
            "context_lens_tensor": Tensor.from_numpy(np.array([0], dtype=np.int32)),
            "block_tables": Tensor.from_numpy(
                np.ones((batch_size, DEFAULT_NUM_BLOCKS)).astype(np.int32)),
            "slot_mapping": Tensor.from_numpy(
                np.ones((batch_size * seq_length,)).astype(np.int32)),
            "logits": Tensor.from_numpy(np.random.randn(batch_size, vocab_size))
        }
    else:
        init_params = {
            "input_ids": Tensor.from_numpy(
                np.random.randint(0, vocab_size, size=(batch_size,), dtype=np.int32)),
            "positions": Tensor.from_numpy(
                np.array([seq_length for _ in range(batch_size)], dtype=np.int32)),
            "attention_mask": None,
            "batch_valid_length": Tensor.from_numpy(
                np.array([seq_length + 1] * batch_size, dtype=np.int32)),
            "q_seq_lens": Tensor.from_numpy(
                np.array([1] * batch_size, dtype=np.int32)),
            "context_lens_tensor": Tensor.from_numpy(
                np.array([seq_length + 1] * batch_size, dtype=np.int32)),
            "block_tables": Tensor.from_numpy(
                np.ones((batch_size, DEFAULT_NUM_BLOCKS)).astype(np.int32)),
            "slot_mapping": Tensor.from_numpy(
                np.ones((batch_size,)).astype(np.int32)),
            "logits": Tensor.from_numpy(np.random.randn(batch_size, vocab_size))
        }

    return init_params


_common_output_shape = (DEFAULT_BATCH_SIZE, DEFAULT_HIDDEN_SIZE)

GOLDEN_DATA = {
    "output_standard": np.random.rand(*_common_output_shape).astype(np.float32),
    "output_qk": np.random.rand(*_common_output_shape).astype(np.float32),
    "output_sandwich": np.random.rand(*_common_output_shape).astype(np.float32),
}


GPU_DATA = {
    "output_standard": np.random.rand(*_common_output_shape).astype(np.float32),
    "output_qk": np.random.rand(*_common_output_shape).astype(np.float32),
    "output_sandwich": np.random.rand(*_common_output_shape).astype(np.float32),
}

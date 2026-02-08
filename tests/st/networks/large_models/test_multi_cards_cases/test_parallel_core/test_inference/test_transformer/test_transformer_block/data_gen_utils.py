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
"""Generate data for mcore Transformer Block UT of inference."""
import numpy as np

DEFAULT_BATCH_SIZE = 2
DEFAULT_SEQ_LENGTH = 2
DEFAULT_HIDDEN_SIZE = 32
DEFAULT_FFN_HIDDEN_SIZE = 64
DEFAULT_NUM_HEADS = 2
DEFAULT_NUM_BLOCKS = 16
DEFAULT_BLOCK_SIZE = 64


def get_init_params(seq_length=DEFAULT_SEQ_LENGTH,
                    batch_size=DEFAULT_BATCH_SIZE,
                    hidden_size=DEFAULT_HIDDEN_SIZE):
    """
    Generates initial parameters (inputs) for the TransformerLayer model.
    """
    np.random.seed(2025)
    hidden_states_shape = (seq_length * batch_size, hidden_size)
    init_params = {
        "hidden_states": 0.01 * np.random.randn(*hidden_states_shape),
    }
    return init_params


_common_output_shape = (DEFAULT_SEQ_LENGTH * DEFAULT_BATCH_SIZE, DEFAULT_HIDDEN_SIZE)

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

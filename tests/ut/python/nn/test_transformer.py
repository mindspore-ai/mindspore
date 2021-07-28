# Copyright 2021 Huawei Technologies Co., Ltd
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
""" test transformer"""
import numpy as np
from mindspore import Tensor
from mindspore.common import dtype
from mindspore.nn.parallel import MultiHeadAttention, FeedForward, TransformerEncoderLayer, TransformerEncoder, \
    TransformerDecoder, TransformerDecoderLayer, Transformer, CrossEntropyLoss, AttentionMask
from mindspore.common.api import _executor


def test_transformer_encoder_only():
    model = Transformer(batch_size=2,
                        src_seq_length=20,
                        tgt_seq_length=0,
                        encoder_layers=2,
                        decoder_layers=0,
                        hidden_size=64,
                        ffn_hidden_size=64)

    encoder_input_value = Tensor(np.ones((2, 20, 64)), dtype.float32)
    encoder_input_mask = Tensor(np.ones((2, 20, 20)), dtype.float16)

    _executor.compile(model, encoder_input_value, encoder_input_mask)


def test_encoder_and_decoder():
    model = Transformer(batch_size=2,
                        src_seq_length=20,
                        tgt_seq_length=10,
                        encoder_layers=1,
                        decoder_layers=2,
                        hidden_size=64,
                        ffn_hidden_size=64)

    encoder_input_value = Tensor(np.ones((2, 20, 64)), dtype.float32)
    encoder_input_mask = Tensor(np.ones((2, 20, 20)), dtype.float16)

    decoder_input_value = Tensor(np.ones((2, 10, 64)), dtype.float32)
    decoder_input_mask = Tensor(np.ones((2, 10, 10)), dtype.float16)
    memory_mask = Tensor(np.ones((2, 10, 20)), dtype.float16)

    _executor.compile(model, encoder_input_value, encoder_input_mask,
                      decoder_input_value,
                      decoder_input_mask,
                      memory_mask)


def test_transformer_encoder():
    model = TransformerEncoder(batch_size=2,
                               seq_length=16,
                               num_layers=2,
                               hidden_size=8,
                               ffn_hidden_size=64,
                               num_heads=2)

    encoder_input_value = Tensor(np.ones((2, 16, 8)), dtype.float32)
    encoder_input_mask = Tensor(np.ones((2, 16, 16)), dtype.float16)

    _executor.compile(model,
                      encoder_input_value,
                      encoder_input_mask)


def test_transformer_encoder_layer():
    model = TransformerEncoderLayer(batch_size=2, hidden_size=8, ffn_hidden_size=64, seq_length=16,
                                    num_heads=2)

    encoder_input_value = Tensor(np.ones((2, 16, 8)), dtype.float32)
    encoder_input_mask = Tensor(np.ones((2, 16, 16)), dtype.float16)

    _executor.compile(model,
                      encoder_input_value,
                      encoder_input_mask)


def test_transformer_encoder_layer_post_ture():
    model = TransformerEncoderLayer(batch_size=2,
                                    seq_length=16,
                                    hidden_size=8, ffn_hidden_size=64,
                                    num_heads=2, post_layernorm_residual=True)

    encoder_input_value = Tensor(np.ones((2, 16, 8)), dtype.float32)
    encoder_input_mask = Tensor(np.ones((2, 16, 16)), dtype.float16)

    _executor.compile(model,
                      encoder_input_value,
                      encoder_input_mask)


def test_transformer_decoder():
    model = TransformerDecoder(num_layers=1,
                               batch_size=2,
                               src_seq_length=20,
                               tgt_seq_length=10,
                               hidden_size=64,
                               ffn_hidden_size=64,
                               num_heads=2)

    encoder_input_value = Tensor(np.ones((2, 20, 64)), dtype.float32)

    decoder_input_value = Tensor(np.ones((2, 10, 64)), dtype.float32)
    decoder_input_mask = Tensor(np.ones((2, 10, 10)), dtype.float16)
    memory_mask = Tensor(np.ones((2, 10, 20)), dtype.float16)

    _executor.compile(model, decoder_input_value, decoder_input_mask,
                      encoder_input_value,
                      memory_mask)


def test_transformer_decoder_layer():
    model = TransformerDecoderLayer(
        batch_size=2,
        src_seq_length=20,
        tgt_seq_length=10,
        hidden_size=64,
        ffn_hidden_size=64,
        num_heads=2)

    encoder_input_value = Tensor(np.ones((2, 20, 64)), dtype.float32)

    decoder_input_value = Tensor(np.ones((2, 10, 64)), dtype.float32)
    decoder_input_mask = Tensor(np.ones((2, 10, 10)), dtype.float16)
    memory_mask = Tensor(np.ones((2, 10, 20)), dtype.float16)

    _executor.compile(model, decoder_input_value, decoder_input_mask,
                      encoder_input_value,
                      memory_mask)


def test_multihead_attention():
    model = MultiHeadAttention(hidden_size=15,
                               src_seq_length=20,
                               tgt_seq_length=20,
                               batch_size=2,
                               num_heads=3)
    from_tensor = Tensor(np.ones((2, 20, 15)), dtype.float32)
    to_tensor = Tensor(np.ones((2, 20, 15)), dtype.float16)
    attention_mask = Tensor(np.ones((2, 20, 20)), dtype.float16)

    _executor.compile(model, from_tensor, to_tensor, to_tensor, attention_mask)


def test_feedforward_layer():
    model = FeedForward(hidden_size=15,
                        ffn_hidden_size=30,
                        dropout_rate=0.1,
                        hidden_act='relu')
    tensor = Tensor(np.ones((2, 20, 15)), dtype.float32)

    _executor.compile(model, tensor)


def test_cross_entroy():
    model = CrossEntropyLoss()
    logits = Tensor(np.array([[3, 5, 6, 9, 12, 33, 42, 12, 32, 72]]), dtype.float32)
    labels_np = np.array([1]).astype(np.int32)
    input_mask = Tensor(np.ones(1).astype(np.float32))
    labels = Tensor(labels_np)
    _executor.compile(model, logits, labels, input_mask)


def test_attention_mask():
    model = AttentionMask(seq_length=19)
    inputs = Tensor(np.ones((2, 19)), dtype.float32)
    _executor.compile(model, inputs)

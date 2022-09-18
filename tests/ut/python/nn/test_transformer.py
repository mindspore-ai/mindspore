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
import os
import shutil
import numpy as np
import pytest


import mindspore
from mindspore import Tensor
from mindspore.common import dtype
from mindspore.parallel.nn import MultiHeadAttention, FeedForward, TransformerEncoderLayer, TransformerEncoder, \
    TransformerDecoder, TransformerDecoderLayer, Transformer, CrossEntropyLoss, AttentionMask, FixedSparseAttention
from mindspore.common.api import _cell_graph_executor


def test_transformer_encoder_only():
    model = Transformer(batch_size=2,
                        src_seq_length=20,
                        tgt_seq_length=10,
                        encoder_layers=2,
                        decoder_layers=0,
                        hidden_size=64,
                        ffn_hidden_size=64)

    encoder_input_value = Tensor(np.ones((2, 20, 64)), dtype.float32)
    encoder_input_mask = Tensor(np.ones((2, 20, 20)), dtype.float16)

    _cell_graph_executor.compile(model, encoder_input_value, encoder_input_mask)


def test_transformer_encoder_log_softmax():
    with pytest.raises(ValueError):
        model = Transformer(batch_size=2,
                            src_seq_length=20,
                            tgt_seq_length=10,
                            encoder_layers=2,
                            decoder_layers=0,
                            hidden_act='logsoftmax',
                            hidden_size=64,
                            ffn_hidden_size=64)

        encoder_input_value = Tensor(np.ones((2, 20, 64)), dtype.float32)
        encoder_input_mask = Tensor(np.ones((2, 20, 20)), dtype.float16)

        _cell_graph_executor.compile(model, encoder_input_value, encoder_input_mask)


def test_transformer_encoder_leakyrelu():
    model = Transformer(batch_size=2,
                        src_seq_length=20,
                        tgt_seq_length=10,
                        encoder_layers=2,
                        decoder_layers=0,
                        hidden_act='leakyrelu',
                        hidden_size=64,
                        ffn_hidden_size=64)

    encoder_input_value = Tensor(np.ones((2, 20, 64)), dtype.float32)
    encoder_input_mask = Tensor(np.ones((2, 20, 20)), dtype.float16)

    _cell_graph_executor.compile(model, encoder_input_value, encoder_input_mask)


def test_transformer_encoder_logsigmoid():
    model = Transformer(batch_size=2,
                        src_seq_length=20,
                        tgt_seq_length=10,
                        encoder_layers=2,
                        decoder_layers=0,
                        hidden_act='logsigmoid',
                        hidden_size=64,
                        ffn_hidden_size=64)

    encoder_input_value = Tensor(np.ones((2, 20, 64)), dtype.float32)
    encoder_input_mask = Tensor(np.ones((2, 20, 20)), dtype.float16)

    _cell_graph_executor.compile(model, encoder_input_value, encoder_input_mask)


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

    _cell_graph_executor.compile(model, encoder_input_value, encoder_input_mask,
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

    _cell_graph_executor.compile(model,
                                 encoder_input_value,
                                 encoder_input_mask)


def test_transformer_encoder_layer():
    model = TransformerEncoderLayer(batch_size=2, hidden_size=8, ffn_hidden_size=64, seq_length=16,
                                    num_heads=2)

    encoder_input_value = Tensor(np.ones((2, 16, 8)), dtype.float32)
    encoder_input_mask = Tensor(np.ones((2, 16, 16)), dtype.float16)

    _cell_graph_executor.compile(model,
                                 encoder_input_value,
                                 encoder_input_mask)


def test_transformer_encoder_layer_post_ture():
    model = TransformerEncoderLayer(batch_size=2,
                                    seq_length=16,
                                    hidden_size=8, ffn_hidden_size=64,
                                    num_heads=2, post_layernorm_residual=True)

    encoder_input_value = Tensor(np.ones((2, 16, 8)), dtype.float32)
    encoder_input_mask = Tensor(np.ones((2, 16, 16)), dtype.float16)

    _cell_graph_executor.compile(model,
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

    _cell_graph_executor.compile(model, decoder_input_value, decoder_input_mask,
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

    _cell_graph_executor.compile(model, decoder_input_value, decoder_input_mask,
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

    _cell_graph_executor.compile(model, from_tensor, to_tensor, to_tensor, attention_mask)


def test_multihead_attention_wrong_batch():
    model = MultiHeadAttention(hidden_size=15,
                               src_seq_length=20,
                               tgt_seq_length=20,
                               batch_size=2,
                               num_heads=3)
    from_tensor = Tensor(np.ones((3, 20, 15)), dtype.float32)
    to_tensor = Tensor(np.ones((3, 20, 15)), dtype.float16)
    attention_mask = Tensor(np.ones((3, 20, 20)), dtype.float16)

    with pytest.raises(ValueError):
        _cell_graph_executor.compile(model, from_tensor, to_tensor, to_tensor, attention_mask)


def test_multihead_attention_fp32_dtype():
    """
    Feature: Test MultiHeadAttention with float32 as compute dtype
    Description: Test using float32 as computation for linear layer.
    Expectation: No exception
    """
    model = MultiHeadAttention(hidden_size=15,
                               src_seq_length=20,
                               tgt_seq_length=20,
                               compute_dtype=dtype.float32,
                               batch_size=2,
                               num_heads=3)
    from_tensor = Tensor(np.ones((2, 20, 15)), dtype.float32)
    to_tensor = Tensor(np.ones((2, 20, 15)), dtype.float32)
    attention_mask = Tensor(np.ones((2, 20, 20)), dtype.float32)
    _cell_graph_executor.compile(model, from_tensor, to_tensor, to_tensor, attention_mask)


def test_feedforward_layer():
    model = FeedForward(hidden_size=15,
                        ffn_hidden_size=30,
                        dropout_rate=0.1,
                        hidden_act='relu')
    tensor = Tensor(np.ones((2, 20, 15)), dtype.float32)

    _cell_graph_executor.compile(model, tensor)


def test_cross_entroy():
    model = CrossEntropyLoss()
    logits = Tensor(np.array([[3, 5, 6, 9, 12, 33, 42, 12, 32, 72]]), dtype.float32)
    labels_np = np.array([1]).astype(np.int32)
    input_mask = Tensor(np.ones(1).astype(np.float32))
    labels = Tensor(labels_np)
    _cell_graph_executor.compile(model, logits, labels, input_mask)


def test_attention_mask():
    model = AttentionMask(seq_length=19)
    inputs = Tensor(np.ones((2, 19)), dtype.float32)
    _cell_graph_executor.compile(model, inputs)


def test_sparse_attention():
    model = FixedSparseAttention(batch_size=2,
                                 seq_length=1024,
                                 size_per_head=64,
                                 num_heads=8,
                                 block_size=64)
    q = Tensor(np.ones((2, 1024, 512)), dtype.float16)
    k = Tensor(np.ones((2, 1024, 512)), dtype.float16)
    v = Tensor(np.ones((2, 1024, 512)), dtype.float16)
    mask = Tensor(np.ones((2, 1024, 1024)), dtype.float32)
    _cell_graph_executor.compile(model, q, k, v, mask)


class TestBasicWarningValidator:
    log_envs = dict(GLOG_v=None, GLOG_logtostderr=None, GLOG_log_dir=None, logger_maxBytes=None,
                    logger_backupCount=None)
    log_path = './TestBasicWarningValidator'

    def setup_method(self):
        for env in self.log_envs:
            self.log_envs[env] = os.environ.get(env, None)
        os.environ['GLOG_log_dir'] = self.log_path
        os.environ['GLOG_v'] = '1'
        os.environ['GLOG_logtostderr'] = '0'
        # Force to generate the logger again
        # pylint: disable=W0212
        mindspore.log.GLOBAL_LOGGER = None

    def teardown_method(self):
        for env in self.log_envs:
            if self.log_envs.get(env, False):
                os.environ[env] = self.log_envs.get(env, "False")
        shutil.rmtree(os.path.join(self.log_path))

    def check_warning_log(self):
        cmd = f'cd {self.log_path} && grep WARNING rank_0/logs/mindspore.log.* |wc -l'
        file_count = os.popen(cmd).read().strip()
        assert file_count == "0"

    def test_cross_entory_no_warning(self):
        """
        Feature: Test the warning log
        Description: Test a forward compile has no warning error
        Expectation: To compile passed
        """
        # Force to rebuild the logger
        test_cross_entroy()
        self.check_warning_log()

    def test_transformer_encoder_no_warning(self):
        """
        Feature: Test the warning log
        Description: Test a forward compile has no warning error
        Expectation: To compile passed
        """
        # Force to rebuild the logger
        test_transformer_encoder_only()
        self.check_warning_log()

    def test_transformer_decoder_no_warning(self):
        """
        Feature: Test the warning log
        Description: Test a forward compile has no warning error
        Expectation: To compile passed
        """
        # Force to rebuild the logger
        test_transformer_decoder()
        self.check_warning_log()

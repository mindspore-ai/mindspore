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
import pytest

import mindspore as ms
from mindspore import Tensor, ops
from mindspore.nn import MultiheadAttention, TransformerEncoderLayer, \
    TransformerEncoder, TransformerDecoderLayer, TransformerDecoder, Transformer
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('dtype', [ms.float16, ms.float32, ms.float64])
@pytest.mark.parametrize('jit', [False, True])
def test_multihead_attention_pynative_cpu_gpu(dtype, jit):
    """
    Feature: MultiheadAttention
    Description: Verify the result of MultiheadAttention
    Expectation: success
    """
    embed_dim = 128
    num_heads = 8
    sl = 10
    bs = 8
    model = MultiheadAttention(embed_dim, num_heads, dtype=dtype)
    q = Tensor(np.random.randn(sl, bs, embed_dim), dtype)
    k = Tensor(np.random.randn(sl, bs, embed_dim), dtype)
    v = Tensor(np.random.randn(sl, bs, embed_dim), dtype)

    def forward(q, k, v):
        out = model(q, k, v)
        return out

    if jit:
        forward = ms.jit(forward)

    out = forward(q, k, v)
    assert q.shape == out[0].shape


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('dtype', [ms.float16, ms.float32])
@pytest.mark.parametrize('jit', [False, True])
def test_multihead_attention_pynative_ascend(dtype, jit):
    """
    Feature: MultiheadAttention
    Description: Verify the result of MultiheadAttention
    Expectation: success
    """
    embed_dim = 128
    num_heads = 8
    sl = 10
    bs = 8
    model = MultiheadAttention(embed_dim, num_heads, dtype=dtype)
    q = Tensor(np.random.randn(sl, bs, embed_dim), dtype)
    k = Tensor(np.random.randn(sl, bs, embed_dim), dtype)
    v = Tensor(np.random.randn(sl, bs, embed_dim), dtype)

    def forward(q, k, v):
        out = model(q, k, v)
        return out

    if jit:
        forward = ms.jit(forward)

    out = forward(q, k, v)
    assert q.shape == out[0].shape


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('dtype', [ms.float16, ms.float32, ms.float64])
@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('jit', [False, True])
def test_transformerencoder_square_input_cpu(dtype, training, jit):
    """
    Feature: TransformerEncoder
    Description: Test for edge cases when input of shape (batch size, sequence length, embedding dimension) has
    batch size == sequence length
    Expectation: success
    """
    model = TransformerEncoder(
        TransformerEncoderLayer(d_model=4, nhead=2, dim_feedforward=16, dropout=0.0, batch_first=True, dtype=dtype),
        num_layers=2)

    # set constant weights of the model
    for _, p in model.parameters_and_names():
        x = p.data
        sz = x.view(-1).shape[0]
        shape = x.shape
        x = ops.cos(ops.arange(0, sz).astype(dtype).view(shape))
        p.set_dtype(dtype)

    if training:
        model = model.set_train()
    else:
        model = model.set_train(False)
    x = ops.arange(0, 16).reshape(2, 2, 4).astype(dtype)
    src_mask = Tensor([[0, 1], [0, 0]]).to(ms.bool_)

    def forward(x, mask):
        result = model(x, src_mask=mask)
        return result

    if jit:
        forward = ms.jit(forward)

    result = forward(x, src_mask)
    ref_output = ms.Tensor([[[2.420306205749512, 0.017629241570830, -0.607857942581177, -0.085519507527351],
                             [2.420306205749512, 0.017629241570830, -0.607857942581177, -0.085519507527351]],
                            [[2.419836044311523, 0.017548924311996, -0.608187675476074, -0.085347734391689],
                             [2.419836044311523, 0.017548924311996, -0.608187675476074, -0.085347734391689]]],
                           dtype)
    assert tuple(result.shape) == tuple(ref_output.shape)
    np.allclose(result.asnumpy(), ref_output.asnumpy(), rtol=1e-7, atol=1e-5)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('dtype', [ms.float16, ms.float32])
@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('jit', [False, True])
def test_transformerencoder_square_input_ascend(dtype, training, jit):
    """
    Feature: TransformerEncoder
    Description: Test for edge cases when input of shape (batch size, sequence length, embedding dimension) has
    batch size == sequence length
    Expectation: success
    """
    model = TransformerEncoder(
        TransformerEncoderLayer(d_model=4, nhead=2, dim_feedforward=16, dropout=0.0, batch_first=True, dtype=dtype),
        num_layers=2)

    # set constant weights of the model
    for _, p in model.parameters_and_names():
        x = p.data
        sz = x.view(-1).shape[0]
        shape = x.shape
        x = ops.cos(ops.arange(0, sz).astype(dtype).view(shape))
        p.set_dtype(dtype)

    if training:
        model = model.set_train()
    else:
        model = model.set_train(False)
    x = ops.arange(0, 16).reshape(2, 2, 4).astype(dtype)
    src_mask = Tensor([[0, 1], [0, 0]]).to(ms.bool_)

    def forward(x, mask):
        result = model(x, src_mask=mask)
        return result

    if jit:
        forward = ms.jit(forward)

    result = forward(x, src_mask)
    ref_output = ms.Tensor([[[2.420306205749512, 0.017629241570830, -0.607857942581177, -0.085519507527351],
                             [2.420306205749512, 0.017629241570830, -0.607857942581177, -0.085519507527351]],
                            [[2.419836044311523, 0.017548924311996, -0.608187675476074, -0.085347734391689],
                             [2.419836044311523, 0.017548924311996, -0.608187675476074, -0.085347734391689]]],
                           dtype)
    assert tuple(result.shape) == tuple(ref_output.shape)
    np.allclose(result.asnumpy(), ref_output.asnumpy(), rtol=1e-7, atol=1e-5)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('dtype', [ms.float16, ms.float32])
@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('jit', [False, True])
def test_transformerencoder_square_input_gpu(dtype, training, jit):
    """
    Feature: TransformerEncoder
    Description: Test for edge cases when input of shape (batch size, sequence length, embedding dimension) has
    batch size == sequence length
    Expectation: success
    """
    model = TransformerEncoder(
        TransformerEncoderLayer(d_model=4, nhead=2, dim_feedforward=16, dropout=0.0, batch_first=True, dtype=dtype),
        num_layers=2)

    # set constant weights of the model
    for _, p in model.parameters_and_names():
        x = p.data
        sz = x.view(-1).shape[0]
        shape = x.shape
        x = ops.cos(ops.arange(0, sz).astype(dtype).view(shape))
        p.set_dtype(dtype)

    if training:
        model = model.set_train()
    else:
        model = model.set_train(False)
    x = ops.arange(0, 16).reshape(2, 2, 4).astype(dtype)
    src_mask = Tensor([[0, 1], [0, 0]]).to(ms.bool_)

    def forward(x, mask):
        result = model(x, src_mask=mask)
        return result

    if jit:
        forward = ms.jit(forward)

    result = forward(x, src_mask)
    ref_output = ms.Tensor([[[2.420306205749512, 0.017629241570830, -0.607857942581177, -0.085519507527351],
                             [2.420306205749512, 0.017629241570830, -0.607857942581177, -0.085519507527351]],
                            [[2.419836044311523, 0.017548924311996, -0.608187675476074, -0.085347734391689],
                             [2.419836044311523, 0.017548924311996, -0.608187675476074, -0.085347734391689]]],
                           dtype)
    assert tuple(result.shape) == tuple(ref_output.shape)
    np.allclose(result.asnumpy(), ref_output.asnumpy(), rtol=1e-7, atol=1e-5)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('dtype', [ms.float16, ms.float32])
@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('jit', [False, True])
def test_transformerdecoder_gpu(dtype, training, jit):
    """
    Feature: TransformerDecoder
    Description: Test shape (batch size, sequence length, embedding dimension)
    Expectation: success
    """
    decoder_layer = TransformerDecoderLayer(d_model=512, nhead=8, dtype=dtype)
    transformer_decoder = TransformerDecoder(decoder_layer, num_layers=6)
    memory = Tensor(np.random.rand(10, 32, 512), dtype)
    tgt = Tensor(np.random.rand(20, 32, 512), dtype)

    if training:
        transformer_decoder.set_train()
    else:
        transformer_decoder.set_train(False)

    def forward(tgt, memory):
        out = transformer_decoder(tgt, memory)
        return out

    if jit:
        forward = ms.jit(forward)

    result = forward(tgt, memory)
    assert result.shape == tgt.shape


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('dtype', [ms.float16, ms.float32, ms.float64])
@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('jit', [False, True])
def test_transformerdecoder_cpu(dtype, training, jit):
    """
    Feature: TransformerDecoder
    Description: Test shape (batch size, sequence length, embedding dimension)
    Expectation: success
    """
    decoder_layer = TransformerDecoderLayer(d_model=512, nhead=8, dtype=dtype)
    transformer_decoder = TransformerDecoder(decoder_layer, num_layers=6)
    memory = Tensor(np.random.rand(10, 32, 512), dtype)
    tgt = Tensor(np.random.rand(20, 32, 512), dtype)

    if training:
        transformer_decoder.set_train()
    else:
        transformer_decoder.set_train(False)

    def forward(tgt, memory):
        out = transformer_decoder(tgt, memory)
        return out

    if jit:
        forward = ms.jit(forward)

    result = forward(tgt, memory)
    assert result.shape == tgt.shape


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('dtype', [ms.float16, ms.float32])
@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('jit', [False, True])
def test_transformerdecoder_ascend(dtype, training, jit):
    """
    Feature: TransformerDecoder
    Description: Test shape (batch size, sequence length, embedding dimension)
    Expectation: success
    """
    decoder_layer = TransformerDecoderLayer(d_model=512, nhead=8, dtype=dtype)
    transformer_decoder = TransformerDecoder(decoder_layer, num_layers=6)
    memory = Tensor(np.random.rand(10, 32, 512), dtype)
    tgt = Tensor(np.random.rand(20, 32, 512), dtype)

    if training:
        transformer_decoder.set_train()
    else:
        transformer_decoder.set_train(False)

    def forward(tgt, memory):
        out = transformer_decoder(tgt, memory)
        return out

    if jit:
        forward = ms.jit(forward)

    result = forward(tgt, memory)
    assert result.shape == tgt.shape


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('dtype', [ms.float16, ms.float32, ms.float64])
@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('jit', [False, True])
def test_transformer_cpu(dtype, training, jit):
    """
    Feature: Transformer
    Description: Test shape (batch size, sequence length, embedding dimension)
    Expectation: success
    """
    transformer_model = Transformer(nhead=16, num_encoder_layers=12, dtype=dtype)
    src = Tensor(np.random.rand(10, 32, 512), dtype)
    tgt = Tensor(np.random.rand(20, 32, 512), dtype)

    if training:
        transformer_model.set_train()
    else:
        transformer_model.set_train(False)

    def forward(src, tgt):
        out = transformer_model(src, tgt)
        return out

    if jit:
        forward = ms.jit(forward)

    result = forward(src, tgt)
    assert result.shape == tgt.shape


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('dtype', [ms.float16, ms.float32])
@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('jit', [False, True])
def test_transformer_gpu(dtype, training, jit):
    """
    Feature: Transformer
    Description: Test shape (batch size, sequence length, embedding dimension)
    Expectation: success
    """
    transformer_model = Transformer(nhead=16, num_encoder_layers=12, dtype=dtype)
    src = Tensor(np.random.rand(10, 32, 512), dtype)
    tgt = Tensor(np.random.rand(20, 32, 512), dtype)

    if training:
        transformer_model.set_train()
    else:
        transformer_model.set_train(False)

    def forward(src, tgt):
        out = transformer_model(src, tgt)
        return out
    if jit:
        forward = ms.jit(forward)

    result = forward(src, tgt)
    assert result.shape == tgt.shape


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('dtype', [ms.float16, ms.float32])
@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('jit', [False, True])
def test_transformer_ascend(dtype, training, jit):
    """
    Feature: Transformer
    Description: Test shape (batch size, sequence length, embedding dimension)
    Expectation: success
    """
    transformer_model = Transformer(nhead=16, num_encoder_layers=12, dtype=dtype)
    src = Tensor(np.random.rand(10, 32, 512), dtype)
    tgt = Tensor(np.random.rand(20, 32, 512), dtype)

    if training:
        transformer_model.set_train()
    else:
        transformer_model.set_train(False)

    def forward(src, tgt):
        out = transformer_model(src, tgt)
        return out

    if jit:
        forward = ms.jit(forward)

    result = forward(src, tgt)
    assert result.shape == tgt.shape

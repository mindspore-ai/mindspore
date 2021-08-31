import numpy as np
from mindspore import Tensor
from mindspore.parallel.nn.layers import FixedSparseAttention
import mindspore.context as context

context.set_context(device_target="Ascend")


def test_net():
    np.random.seed(0)
    bs = 2  # batch size
    heads = 2
    seq_len = 1024  # this op is designed for seq_len = 1024
    size_per_head = 128  # maximum size per head value is 128

    block_size = 64  # block size is designed to be 64
    fixed_sparse = FixedSparseAttention(bs, heads, size_per_head, block_size)
    q = np.random.rand(bs, seq_len, heads * size_per_head)
    q = q.astype(np.float16)
    k = np.random.rand(bs, seq_len, heads * size_per_head)
    k = k.astype(np.float16)
    v = np.random.rand(bs, seq_len, heads * size_per_head)
    v = v.astype(np.float16)
    attention_mask = np.ones((bs, seq_len, seq_len), dtype=np.float32)
    out = fixed_sparse(Tensor(q), Tensor(k), Tensor(v), Tensor(attention_mask))
    out_np = out.asnumpy()
    print("local output: ", out_np[0, 0])

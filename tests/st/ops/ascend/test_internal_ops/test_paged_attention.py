import mindspore as ms
import numpy as np
import math


def gen_seq_len(batch, max_seq, pa=False, rand = False):
    if pa:
        max_seq_aligned = 16
        seqlen = np.ones((batch,)).astype(np.int32)
        print(seqlen)
        seqlen_aligned = np.ones((batch,)) * max_seq_aligned
        seqlen_aligned = seqlen_aligned.astype(np.int32)
    elif rand:
        max_seq_aligned = (max_seq + 15) // 16 * 16
        randnum = np.random.randint(1, max_seq, 1)
        seqlen = np.ones((batch,)) * randnum
        seqlen = seqlen.astype(np.int32)
        print("seqlen:",seqlen)
        seqlen_aligned = np.ones((batch,)) * max_seq_aligned
        seqlen_aligned = seqlen_aligned.astype(np.int32)
    else:
        max_seq_aligned = (max_seq + 15) // 16 * 16
        seqlen = np.ones((batch,)) * max_seq
        seqlen = seqlen.astype(np.int32)
        print(seqlen)
        seqlen_aligned = np.ones((batch,)) * max_seq_aligned
        seqlen_aligned = seqlen_aligned.astype(np.int32)
    
    ntokens = seqlen.sum()
    print("ntokens:", ntokens)
    return seqlen, seqlen_aligned, ntokens

def calc_expect_func(batch, max_seq, q_heads, kv_heads, embed, drop_prop, low_tri=True, is_dropout=False,\
    pa = False, inc = False, input_type='float16', calc_type = np.float32, layout = 'BNSD', block_size = 16):
    num_groups = q_heads // kv_heads
    q_seqlen, q_seqlen_aligned, q_ntokens = gen_seq_len(batch, max_seq, pa | inc)
    if pa:
        kv_seqlen, kv_seqlen_aligned, kv_ntokens = gen_seq_len(batch, max_seq, False)
    else:
        kv_seqlen, kv_seqlen_aligned, kv_ntokens = gen_seq_len(batch, max_seq, False)
    np.random.seed(0)
    ntokens2 = np.array(max_seq * max_seq)

    q = np.random.uniform(-1.0, 1.0, size=(batch, q_heads, q_seqlen[0], embed)).astype(np.float16)
    k = np.random.uniform(-1.0, 1.0, size=(batch, kv_heads, max_seq, embed)).astype(np.float16)
    v = np.random.uniform(-1.0, 1.0, size=(batch, kv_heads, max_seq, embed)).astype(np.float16)
    dropout_uint8 = np.zeros(batch * max_seq * max_seq >> 3).astype(np.uint8)

    amask = np.ones(shape=(batch, max_seq, max_seq)).astype(np.float16)
    amask = np.triu(amask, 1)  #下三角
    dmask = None
    if is_dropout:
        dmask = np.random.uniform(size=(batch * max_seq * max_seq)) > drop_prop
        masks = (0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80)
        j = 0
        for i in range(batch * max_seq * max_seq >> 3):
            for m in masks:
                if dmask[j]:
                    dropout_uint8[i] = dropout_uint8[i] | m
                j += 1
        dmask = dmask.reshape(batch, max_seq, max_seq)

    q_offset = 0
    k_offset = 0
    v_offset = 0

    s = None
    _p = None
    out = None
    _score_max = None
    _score_sum = None

    for idx in range(batch):
        for hidx in range(q_heads):
            kv_hidx = hidx // num_groups
            q_s = q_seqlen[idx]
            kv_s = kv_seqlen[idx]
            q_slice = q[idx,hidx,:,:]
            k_slice = k[idx,kv_hidx,0:kv_seqlen[0],:]
            k_slice_t = np.transpose(k_slice, (1, 0))   # get K^T 
            v_slice = v[idx,kv_hidx,0:kv_seqlen[0],:]
            if is_dropout:
                dmask_slice = dmask[idx,:,:]
            if low_tri:
                amask_slice = amask[idx,:,:]
            score = np.matmul(q_slice.astype(np.float32),
                        k_slice_t.astype(np.float32)).astype(np.float16)
            if s is None:
                s = score.reshape([-1,])
            else:
                s = np.concatenate((s, score.reshape([-1,])), 0)

            tor = np.float16(math.sqrt(1.0 * embed))
            score = score / tor
            if low_tri:
                for i in range(q_s):
                    score[i][:] = score[i][:] - amask_slice[i][:] * 10000
            score_max = np.max(score, axis=-1)
            if _score_max is None:
                _score_max = score_max.astype(np.float16).reshape([-1,])
            else:
                _score_max = np.concatenate((_score_max, score_max.astype(np.float16).reshape([-1,])), 0)
            score = score - score_max.reshape((q_s, 1))
            score_exp = np.exp(score.astype(np.float32))
            
            if is_dropout:
                score_exp = score_exp * dmask_slice / (1-drop_prop)

            score_sum = np.sum(score_exp.astype(np.float16), axis=-1)
            if _score_sum is None:
                _score_sum = score_sum.astype(np.float16).reshape([-1,])
            else:
                _score_sum = np.concatenate((_score_sum, score_sum.astype(np.float16).reshape([-1,])), 0)
            
            p = score_exp.astype(np.float16) / score_sum.reshape((q_s, 1)).astype(np.float16)

            if _p is None:
                _p = p.astype(np.float16).reshape([-1,])
            else:
                _p = np.concatenate((_p, p.astype(np.float16).reshape([-1,])), 0)
            
            o = np.matmul(p.astype(np.float32),
                        v_slice.astype(np.float32)).astype(np.float16)
            o = o.reshape(q_s, embed)
            o = np.ascontiguousarray(o)

            if out is None:
                out = o
            else:
                out = np.concatenate((out, o), 0)

            q_offset += q_s
            k_offset += max_seq
            v_offset += max_seq

    table = np.reshape(np.zeros(batch * max_seq // block_size), (batch, max_seq // block_size))
    
    if layout == "BSH":
        q_bsh = np.reshape(np.zeros(batch * q_seqlen[0] * embed * q_heads), (batch, q_seqlen[0], q_heads * embed))
        k_bsh = np.reshape(np.zeros(batch * max_seq * embed * kv_heads), (batch, max_seq, kv_heads * embed))
        v_bsh = np.reshape(np.zeros(batch * max_seq * embed * kv_heads), (batch, max_seq, kv_heads * embed))

        for bidx in range(batch):
            for sidx in range(q_seqlen[0]):
                for hidx in range(embed * q_heads):
                    q_bsh[bidx][sidx][hidx] = q[bidx][hidx // embed][sidx][hidx % embed]
            for sidx in range(max_seq):
                for hidx in range(embed * kv_heads):
                    k_bsh[bidx][sidx][hidx] = k[bidx][hidx // embed][sidx][hidx % embed]
                    v_bsh[bidx][sidx][hidx] = v[bidx][hidx // embed][sidx][hidx % embed]
            if pa:
                for sidx in range(max_seq // block_size):
                        table[bidx][sidx] = bidx * max_seq * kv_heads * embed\
                                          + sidx * block_size * kv_heads * embed

    return q_bsh, k_bsh, v_bsh, table, kv_seqlen, out


class PagedAttention(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.paged_attention = ms.ops.PagedAttention(head_num=8, kv_head_num=1)
 
    def construct(self, query, key_cache, value_cache, block_tables, context_lens):
        output = self.paged_attention(query, key_cache, value_cache, block_tables, context_lens)
        return output


def test_internel_paged_attention_bfloat16():
    """
    Feature: test internal paged attention with bf16
    Description: test internal paged attention with bf16
    Expectation: the result is correct
    """
    ms.set_context(mode=ms.GRAPH_MODE)
    
    batch = 2
    q_heads = 88
    kv_heads = 8
    max_seq = 4096
    embed = 128
    drop_prop = 0.5
    q, k, v, table, kv_seqlen, out = \
    calc_expect_func(batch, max_seq, q_heads, kv_heads, embed, drop_prop, \
        low_tri=False, is_dropout=False, pa=True, input_type='float32', layout = "BSH")

    query = ms.Tensor(q, ms.bfloat16)
    key_cache = ms.Tensor(k, ms.bfloat16)
    value_cache = ms.Tensor(v, ms.bfloat16)
    block_tables = ms.Tensor(table, ms.uint64)
    context_lens = ms.Tensor(kv_seqlen, ms.uint64)

    pa_net = PagedAttention()
    output = pa_net(query, key_cache, value_cache, block_tables, context_lens)
    output = output.reshape(-1).astype(ms.float32).numpy()
    expect = out.astype(np.float32).reshape(-1)

    count = 0
    for index in range(0, len(output)):
        if np.abs(output[index] - expect[index]) > np.abs(expect[index]) * 0.03:
            count = count + 1

    err_ratio = count / len(output)
    print("err_ratio--",err_ratio)
    assert err_ratio < 0.06

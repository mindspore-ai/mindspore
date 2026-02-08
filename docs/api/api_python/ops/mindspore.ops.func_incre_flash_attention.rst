mindspore.ops.incre_flash_attention
===================================

.. py:function:: mindspore.ops.incre_flash_attention(query, key, value, attn_mask=None, actual_seq_lengths=None, pse_shift=None, \
                                                     dequant_scale1=None, quant_scale1=None, dequant_scale2=None, quant_scale2=None, \
                                                     quant_offset2=None, antiquant_scale=None, antiquant_offset=None, block_table=None, \
                                                     num_heads=1, input_layout='BSH', scale_value=1.0, num_key_value_heads=0, \
                                                     block_size=0, inner_precise=1, kv_padding_size=None)

    增量推理场景接口。

    - b：batch维
    - N：注意力头数
    - kvN： `key` / `value` 头数
    - S：序列长度
    - D：头维度
    - H：隐藏层大小
    - kvH： `key` / `value` 的隐藏层大小

    其中 :math:`H=N\times D` ， :math:`kvH=kvN\times D` 。

    self-attention（自注意力）利用输入样本自身的关系构建了一种注意力模型。其原理是假设有一个长度为 :math:`n` 的输入样本序列 :math:`x` ， :math:`x` 的每个元素都是一个 :math:`d` 维向量，
    可以将每个 :math:`d` 维向量看作一个token embedding，将这样一条序列经过3个权重矩阵变换得到3个维度为 :math:`n\times d` 的矩阵。

    self-attention的计算公式一般定义如下，

    .. math::
        Attention(Q,K,V)=Softmax(\frac{QK^{T} }{\sqrt{d} } )V

    其中 :math:`Q` 和 :math:`K^{T}` 的乘积代表输入 :math:`x` 的注意力，为避免该值变得过大，通常除以 :math:`d` 的平方根进行缩放，并对每行进行softmax归一化，与 :math:`V` 相乘后得到一个 :math:`n\times d` 的矩阵。

    .. note::
        - 如果没有输入参数且没有默认值，需要传入 ``None`` 。
        - 参数 `key` 和 `value` 对应的Tensor的shape需要完全一致。
        - 参数 `query` 的 :math:`N` 等于 `num_heads` ，而参数 `key` 和 `value` 的 :math:`N` 等于 `num_key_value_heads` 。 `num_heads` 必须是 `num_key_value_heads` 的倍数。

        - 量化

          - 当 `query`、 `key` 和 `value` 的数据类型为float16，且输出为int8时，需要传入 `quant_scale2`， `quant_offset2` 可选。
          - 当 `antiquant_scale` 存在时， `key` 和 `value` 需要以int8传入， `antiquant_offset` 可选。
          - `antiquant_scale` 和 `antiquant_offset` 的数据类型需要与 `query` 一致。

        - `pse_shift`：

          - `pse_shift` 的数据类型需与 `query` 一致，并且仅支持D轴对齐，即D轴可以被16整除。

        - Page attention：

          - 开启page attention的必要条件是 `block_table` 存在，并且 `key` 和 `value` 根据 `block_table` 中的索引在内存中连续排列。
            支持的 `key` 和 `value` 数据类型为float16/bfloat16/int8。
          - 在page attention开启场景下，当 `key` 和 `value` 的输入类型为float16/bfloat16时，需要16对齐；
            当输入类型为int8时，需要32对齐，推荐使用128对齐。
          - `block_table` 当前支持的max_block_num_per_seq最大为16k，超过16k会导致拦截和报错；
            如果 :math:`S` 过大导致 `max_block_num_per_seq` 超过16k，可通过增大 `block_size` 来解决。
          - 在page attention场景中， `key` 和 `value` 的Tensor的shape各维乘积不能超过int32的表示范围。
          - 在进行per-channel后量化时，不能同时启用page attention。

        - `kv_padding_size`：

          - KV缓存传输起点的计算公式为 :math:`S-kv\_padding\_size-actual\_seq\_lengths`，
            KV缓存传输终点的计算公式为 :math:`S-kv\_padding\_size` 。当起点或终点小于0时，返回的数据结果为全0。
          - 当 `kv_padding_size` 小于0时，会设置为0。
          - `kv_padding_size` 需要与 `actual_seq_lengths` 参数一同启用，否则被认为是KV右填充场景。
          - 需要与 `attn_mask` 参数一同启用，并确保 `attn_mask` 的意义正确，即能正确隐藏无效数据，否则会引入精度问题。
          - `kv_padding_size` 不支持page attention场景。

    .. warning::
        只支持 `Atlas A2` 训练系列产品。

    参数：
        - **query** (Tensor) - 公式中的输入Q，数据类型float16或bfloat16。shape为 :math:`(B, 1, H)` / :math:`(B, N, 1, D)` 。
        - **key** (Union[tuple, list]) - 公式中的输入K，数据类型为float16、bfloat16或int8。shape为 :math:`(B, S, kvH)` / :math:`(B, kvN, S, D)` 。
        - **value** (Union[tuple, list]) - 公式中的输入V，数据类型为float16、bfloat16或int8。shape为 :math:`(B, S, kvH)` / :math:`(B, kvN, S, D)` 。
        - **attn_mask** (Tensor，可选) - 注意力掩码Tensor，数据类型为bool、int8或uint8。shape为 :math:`(B, S)` / :math:`(B, 1, S)` / :math:`(B, 1, 1, S)` 。默认 ``None`` 。
        - **actual_seq_lengths** (Union[Tensor, tuple[int], list[int]]，可选) - 描述每个输入的实际序列长度，数据类型为int64。shape为 :math:`(B, )` 。默认值: ``None`` 。
        - **pse_shift** (Tensor，可选) - 位置编码Tensor，数据类型为float16或bfloat16。输入Tensor的shape为 :math:`(1, N, 1, S)` / :math:`(B, N, 1, S)` 。默认值: ``None`` 。
        - **dequant_scale1** (Tensor，可选) - 量化参数，数据类型为uint64或float32。当前未使能。默认值: ``None`` 。
        - **quant_scale1** (Tensor，可选) - 量化参数，数据类型为float32。当前未使能。默认值: ``None`` 。
        - **dequant_scale2** (Tensor，可选) - 量化参数，数据类型为uint64或float32。当前未使能。默认值: ``None`` 。
        - **quant_scale2** (Tensor，可选) - 后量化参数，数据类型为float32。shape为 :math:`(1,)` 。默认值: ``None`` 。
        - **quant_offset2** (Tensor，可选) - 后量化参数，数据类型为float32。shape为 :math:`(1,)` 。默认值: ``None`` 。
        - **antiquant_scale** (Tensor，可选) - 伪量化参数，数据类型为float16或bfloat16。当 `input_layout` 为 `'BNSD'` 时，shape为 :math:`(2, kvN, 1, D)`；当 `input_layout` 为 `'BSH'` 时，shape为 :math:`(2, kvH)` 。默认值: ``None`` 。
        - **antiquant_offset** (Tensor，可选) - 伪量化参数，数据类型为float16或bfloat16。当 `input_layout` 为 `'BNSD'` 时，shape为 :math:`(2, kvN, 1, D)`；当 `input_layout` 为 `'BSH'` 时，shape为 :math:`(2, kvH)` 。默认值: ``None`` 。
        - **block_table** (Tensor，可选) - Tensor，数据类型为int32。shape为 :math:`(B, max\_block\_num\_per\_seq)`，其中 :math:`max\_block\_num\_per\_seq = ceil(\frac{max(actual\_seq\_length)}{block\_size} )` 。默认值: ``None`` 。
        - **num_heads** (int，可选) - 头的数量。默认值: ``1`` 。
        - **input_layout** (str，可选) - 输入qkv的数据布局，支持 ``'BSH'`` 和 ``'BNSD'`` 。默认值: ``'BSH'`` 。
        - **scale_value** (double，可选) - 表示缩放系数的值，用作计算中的标量。默认值: ``1.0`` 。
        - **num_key_value_heads** (int，可选) - 用于GQA算法的 `key` / `value` 头数。如果 `key` 和 `value` 具有相同的头数，则值为0，使用 `num_heads` 。默认值: ``0`` 。
        - **block_size** (int，可选) - 页注意力中存储在每个KV块中的最大标记数。默认值: ``0`` 。
        - **inner_precise** (int，可选) - 一个 {0, 1} 中的整数，指定计算模式。 ``0`` 为高精度模式（适用于float16 数据类型）， ``1`` 为高性能模式。默认值为 ``1`` 。
        - **kv_padding_size** (Tensor，可选) - Tensor，数据类型为int64。值的范围为 :math:`0\le kv\_padding\_size \le  S-max(actual\_seq\_length)` 。shape为 :math:`()` 或 :math:`(1,)` 。默认值: ``None`` 。

    返回：
        attention_out (Tensor)，注意力输出Tensor，shape为 :math:`(B, 1, H)` / :math:`(B, N, 1, D)` 。

    异常：
        - **TypeError** - `query` 的数据类型不是float16或bfloat16。
        - **TypeError** - `key` 和 `value` 的数据类型不同。
        - **TypeError** - `attn_mask` 的数据类型不是bool、int8或uint8。
        - **TypeError** - `pse_shift` 的数据类型不是bfloat16或float16。
        - **TypeError** - `scale_value` 不是double类型。
        - **TypeError** - `input_layout` 不是字符串。
        - **TypeError** - `num_key_value_heads` 或 `num_heads` 不是整数。
        - **TypeError** - `inner_precise` 不是整数。
        - **TypeError** - `quant_scale1` 不是float32类型的Tensor。
        - **TypeError** - `quant_scale2` 不是float32类型的Tensor。
        - **TypeError** - `quant_offset2` 不是float32类型的Tensor。
        - **ValueError** - `actual_seq_lengths` 的长度不是1或者B。
        - **ValueError** - `input_layout` 是字符串但不是BSH或BNSD。
        - **ValueError** - `num_heads` 不能被Q_H整除。
        - **ValueError** - `num_heads` 不能被 `num_key_value_heads` 整除。
        - **RuntimeError** - `num_heads` 小于等于 0。
        - **RuntimeError** - `attn_mask` 的shape不合法。

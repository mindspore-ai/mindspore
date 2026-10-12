mindspore.ops.prompt_flash_attention
====================================

.. py:function:: mindspore.ops.prompt_flash_attention(query, key, value, attn_mask=None, actual_seq_lengths=None, actual_seq_lengths_kv=None,\
                                                      pse_shift=None, deq_scale1=None, quant_scale1=None, deq_scale2=None, quant_scale2=None,\
                                                      quant_offset2=None, num_heads=1, scale_value=1.0, pre_tokens=2147483647, next_tokens=0,\
                                                      input_layout='BSH', num_key_value_heads=0, sparse_mode=0, inner_precise=1)

    全量推理场景接口。

    - B：batch维
    - N：注意力头数
    - S：序列长度
    - D：头维度
    - H：隐藏层大小

    self-attention（自注意力）利用输入样本自身的关系构建了一种注意力模型。其原理是假设有一个长度为 :math:`n` 的输入样本序列 :math:`x` ， :math:`x` 的每个元素都是一个 :math:`d` 维向量，
    可以将每个 :math:`d` 维向量看作一个token embedding，将这样一条序列经过3个权重矩阵变换得到3个维度为 :math:`n\times d` 的矩阵。

    self-attention的计算公式一般定义如下，

    .. math::
        Attention(Q,K,V)=Softmax(\frac{QK^{T} }{\sqrt{d} } )V

    其中 :math:`Q` 和 :math:`K^{T}` 的乘积代表输入 :math:`x` 的注意力，为避免该值变得过大，通常除以 :math:`d` 的平方根进行缩放，并对每行进行softmax归一化，与 :math:`V` 相乘后得到一个 :math:`n\times d` 的矩阵。

    .. warning::
        - 未来将不再支持 `attn_mask` 的float16数据类型。
        - 当 `sparse_mode` 为2、3或4时， `attn_mask` 的shape必须为 :math:`(2048, 2048)` / :math:`(B, 1, 2048, 2048)` / :math:`(1, 1, 2048, 2048)` 。

    .. note::
        - 各轴的最大支持值：

          - 支持B轴值小于等于65536(64k)。当输入类型包括D轴未对齐到32的int8或float16或bfloat16时，B轴支持最多为128。
          - 支持N轴值小于等于256。
          - 支持S轴值小于等于20971520(20M)。
          - 支持D轴值小于等于512。

        - 量化：

          - int8输入，int8输出：必须提供 `deq_scale1`、 `quant_scale1`、 `deq_scale2` 和 `quant_scale2` 参数。 `quant_offset2` 是可选的（如果未提供，默认值为0）。
          - int8输入，float16输出：必须提供 `deq_scale1`、 `quant_scale1` 和 `deq_scale2` 参数。如果提供 `quant_offset2` 或 `quant_scale2`（不为空），将导致错误。
          - float16或bfloat16输入，int8输出：必须提供 `quant_scale2` 参数。 `quant_offset2` 是可选的（如果未提供，默认值为0）。如果提供 `deq_scale1`、 `quant_scale1` 或 `deq_scale2` 参数（不为空），将导致错误。
          - int8输出：

            - per-channel格式的 `quant_scale2` 和 `quant_offset2` 不支持有左填充、Ring Attention或D轴未对齐到32字节的情况。
            - 不支持带有负值的带状稀疏和 `pre_tokens`/ `next_tokens` 。

          - `quant_scale2` 和 `quant_offset2` 只有在 `query` 为bfloat16时才可以是bfloat16。

        - 其他使用注意事项：

          - `query` 参数的 `N` 必须等于 `num_heads` 。 `key` 和 `value` 参数的 `N` 必须等于 `num_key_value_heads` 。
          - `num_heads` 必须可以被 `num_key_value_heads` 整除，并且 `num_heads` 除以 `num_key_value_heads` 不得大于64。
          - 当 `query` 数据类型为bfloat16时，D轴应与16对齐。
          - `actual_seq_lengths` 的每个元素不得超过q_S， `actual_seq_lengths_kv` 的每个元素不得超过kv_S。

    参数：
        - **query** (Tensor) - 公式中的输入Q，数据类型可以是int8、float16或bfloat16。shape为 :math:`(B, q_S, q_H)` 或 :math:`(B, q_N, q_S, q_D)` 。
        - **key** (Tensor) - 公式中的输入K，数据类型与 `query` 相同。shape为 :math:`(B, kv_S, kv_H)` 或 :math:`(B, kv_N, kv_S, kv_D)` 。
        - **value** (Tensor) - 公式中的输入V，数据类型与 `query` 相同。shape为 :math:`(B, kv_S, kv_H)` 或 :math:`(B, kv_N, kv_S, kv_D)` 。
        - **attn_mask** (Tensor，可选) - 注意力掩码Tensor，数据类型为bool、int8、uint8或float16。每个元素中，0/False表示保留，1/True表示丢弃。如果 `sparse_mode` 为0或1，其shape可以是 :math:`(q_S, kv_S)`、 :math:`(B, q_S, kv_S)`、 :math:`(1, q_S, kv_S)`、 :math:`(B, 1, q_S, kv_S)` 或 :math:`(1, 1, q_S, kv_S)` 。如果 `sparse_mode` 为2、3或4，其shape应为 :math:`(2048, 2048)`、 :math:`(1, 2048, 2048)` 或 :math:`(1, 1, 2048, 2048)` 。默认值为 ``None`` 。
        - **actual_seq_lengths** (Union[Tensor, tuple[int], list[int]]，可选) - 描述 `query` 每个批次的实际序列长度，数据类型为int64。shape为 :math:`(B,)` ，每个元素应为正整数。默认值为 ``None`` 。
        - **actual_seq_lengths_kv** (Union[Tensor, tuple[int], list[int]]，可选) - 描述 `key` 或 `value` 每个批次的实际序列长度，数据类型为int64。shape为 :math:`(B,)` ，每个元素应为正整数。默认值为 ``None`` 。
        - **pse_shift** (Tensor，可选) - 位置编码Tensor，数据类型为float16或bfloat16。输入Tensor shape为 :math:`(B, N, q_S, kv_S)` 或 :math:`(1, N, q_S, kv_S)` 。默认值为 ``None`` 。

          - q_S必须大于等于query的S长度，kv_S必须大于等于key的S长度。
          - 如果 `pse_shift` 的数据类型为float16， `query` 应为float16或int8，这种情况下会自动启用高精度模式。
          - 如果 `pse_shift` 的数据类型为bfloat16， `query` 应为bfloat16。

        - **deq_scale1** (Tensor，可选) - 量化参数，数据类型为uint64或float32。输入Tensor shape为 :math:`(1,)` 。默认值为 ``None`` 。
        - **quant_scale1** (Tensor，可选) - 量化参数，数据类型为float32。输入Tensor shape为 :math:`(1,)` 。默认值为 ``None`` 。
        - **deq_scale2** (Tensor，可选) - 量化参数，输入Tensor shape为 :math:`(1,)` 并与 `deq_scale1` 类型相同。默认值为 ``None`` 。
        - **quant_scale2** (Tensor，可选) - 量化参数，数据类型为float32或bfloat16。当输出layout为BSH时，建议shape为 :math:`(1,)`、 :math:`(1, 1, q_H)`、 :math:`(q_H,)` ；当layout为BNSD时，建议shape为 :math:`(1,)`、 :math:`(1, q_N, 1, D)`、 :math:`(q_N, D)` 。默认值为 ``None`` 。
        - **quant_offset2** (Tensor，可选) - 量化参数，数据类型为float32或bfloat16。数据类型和shape与 `quant_scale2` 相同。默认值为 ``None`` 。
        - **num_heads** (int，可选) - 头的数量，范围为[0, 256]。默认值为 ``1`` 。
        - **scale_value** (double，可选) - 表示缩放系数的值，用作计算中的乘数标量。默认值为 ``1.0`` 。
        - **pre_tokens** (int，可选) - 用于稀疏计算，表示注意力需要关联的前序列元素个数。默认值为 ``2147483647``。
        - **next_tokens** (int，可选) - 用于稀疏计算，表示注意力需要关联的后序列元素个数。默认值为 ``0``。
        - **input_layout** (str，可选) - 输入qkv的数据layout，支持 `BSH` 和 `BNSD` 。默认值为 ``BSH`` 。
        - **num_key_value_heads** (int，可选) - 一个整数，表示在GQA算法中 `key` 和 `value` 的头数量。0表示 `key` 和 `value` 具有与 `query` 相同的头数。如果指定（非0），其必须是 `num_heads` 的因子，并且等于kv_n。默认值为 ``0`` 。
        - **sparse_mode** (int，可选) - 一个整数，指定稀疏模式，可以是 {0, 1, 2, 3, 4} 中的值。默认值为 ``0`` 。

          - sparseMode = 0：如果 `attn_mask` 为空指针，则忽略 `pre_tokens` 和 `next_tokens` 输入（内部设置为INT_MAX）。
          - sparseMode = 2, 3, 4： `attn_mask` shape必须为 :math:`(S, S)`、 :math:`(1, S, S)` 或 :math:`(1, 1, S, S)` ，S固定为2048。用户必须确保 `attn_mask` 为下三角。shape不正确或未提供将导致错误。
          - sparseMode = 1, 2, 3：忽略 `pre_tokens` 和 `next_tokens` 输入，并根据特定规则设置值。
          - sparseMode = 4： `pre_tokens` 和 `next_tokens` 必须是非负的。

        - **inner_precise** (int，可选) - 一个 {0, 1} 中的整数，指定计算模式。 ``0`` 为高精度模式（适用于float16 数据类型）， ``1`` 为高性能模式。默认值为 ``1`` 。

    返回：
        - **attention_out** (Tensor) - 输出Tensor，与 `query` 的shape相同： `(B, q_S, q_H)` 或 `(B, q_N, q_S, q_D)`。输出数据类型由多种因素决定，请参阅上文说明部分获取详细信息。

    异常：
        - **TypeError** - `query` 的数据类型不是int8、float16或bfloat16。
        - **TypeError** - `query`、 `key` 和 `value` 的数据类型不同。
        - **TypeError** - `attn_mask` 的数据类型不是bool、int8或uint8。
        - **TypeError** - `pse_shift` 的数据类型不是bfloat16或float16。
        - **TypeError** - `scale_value` 不是double类型。
        - **TypeError** - `input_layout` 不是字符串。
        - **TypeError** - `num_key_value_heads` 不是整数。
        - **TypeError** - `sparse_mode` 不是整数。
        - **TypeError** - `inner_precise` 不是整数。
        - **TypeError** - `quant_scale1` 不是float32类型的Tensor。
        - **TypeError** - `deq_scale1` 不是uint64或float32类型的Tensor。
        - **TypeError** - `quant_scale2` 不是float32类型的Tensor。
        - **TypeError** - `deq_scale2` 不是uint64或float32类型的Tensor。
        - **TypeError** - `quant_offset2` 不是float32类型的Tensor。
        - **ValueError** - `input_layout` 是字符串但不是BSH或BNSD。
        - **RuntimeError** - `num_heads` 不能被 `num_key_value_heads` 整除。
        - **RuntimeError** - `num_heads` 小于等于 0。
        - **RuntimeError** - `num_key_value_heads` 小于等于0。
        - **RuntimeError** - kv_n不等于 `num_key_value_heads`。
        - **RuntimeError** - `attn_mask` 的shape不合法。
        - **RuntimeError** - `sparse_mode` 被指定的值不是0、1、2、3或4。
        - **RuntimeError** - `query` 的数据类型为bfloat16并且D轴未对齐到16。
        - **RuntimeError** - 输入layout为BSH并且kv_h不能被 `num_key_value_heads` 整除。
        - **RuntimeError** - `query`、 `key` 和 `value` 的D轴不相同。
        - **RuntimeError** - 后量化per-channel情况下，D轴未对齐到32字节。

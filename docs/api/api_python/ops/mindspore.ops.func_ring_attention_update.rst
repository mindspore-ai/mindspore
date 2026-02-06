mindspore.ops.ring_attention_update
===================================

.. py:function:: mindspore.ops.ring_attention_update(prev_attn_out, prev_softmax_max, prev_softmax_sum, cur_attn_out, cur_softmax_max, cur_softmax_sum, actual_seq_qlen=None, layout="SBH")

    RingAttentionUpdate算子功能是将两次FlashAttention的输出根据其不同的softmax的max和sum更新。

    - S：序列长度
    - B：batch维
    - H：隐藏层大小，等于N*D
    - T：time，等于B*S
    - N：注意力头数
    - D：头维度

    .. warning::
        - 仅支持 Atlas A2 训练系列产品。
        - 这是一个实验性API，后续可能修改或删除。
        - 当 `layout` 为 ``"TND"`` 时， `prev_attn_out` 的最后一个维度需要为64的倍数。
        - 当 `layout` 为 ``"TND"`` 时， `actual_seq_qlen` 为必填。
        - 当 `layout` 为 ``"TND"`` 时，输入Shape不能过大。需要注意N*D的大小，限制为： 
          :math:`(\text{AlignUp}(N*D, 64)*(DataSize*6+8))+(\text{AlignUp}(N*8, 64)*56) <= 192*1024`。
          `prev_attn_out` 数据类型为float32时，:math:`DataSize` 为4；数据类型为float16 / bfloat16时，:math:`DataSize` 为2。
        - 当 `layout` 为 ``"TND"`` 时，如果 `actual_seq_qlen` 不是从0到T的非递减序列，计算结果未定义。

    参数：
        - **prev_attn_out** (Tensor) - 第一次FlashAttention的输出，数据类型支持float16、float32、bfloat16，输入shape为 :math:`(S, B, H)` 或 :math:`(T, N, D)`。
        - **prev_softmax_max** (Tensor) - 第一次FlashAttention的softmax的max结果，数据类型支持float32，输入shape为 :math:`(B,N,S,8)` 或 :math:`(T,N,8)`，最后一维8个数字相同，且需要为正数。
        - **prev_softmax_sum** (Tensor) - 第一次FlashAttention的softmax的sum结果，数据类型与shape和 `prev_softmax_max` 保持一致。
        - **cur_attn_out** (Tensor) - 第二次FlashAttention的输出，数据类型与shape和 `prev_attn_out` 保持一致。
        - **cur_softmax_max** (Tensor) - 第二次FlashAttention的softmax的max结果，数据类型与shape和 `prev_softmax_max` 保持一致。
        - **cur_softmax_sum** (Tensor) - 第二次FlashAttention的softmax的sum结果，数据类型与shape和 `prev_softmax_max` 保持一致。
        - **actual_seq_qlen** (Tensor，可选) - 从0开始的sequence length的累加，数据类型支持int64。当数据排布 `layout` 为 ``"TND"`` 时，需要传入该参数；当 `layout` 为 ``"SBH"`` 时，该参数不生效。需要是一个从0开始非递减至T的整数1D Tensor。默认值： ``None``。
        - **layout** (str，可选) - attn_out相关输入的数据排布。当前支持 ``"TND"`` 和 ``"SBH"`` 。默认值： ``"SBH"`` 。

    返回：
        tuple (Tensor)，表示3个Tensor组成的tuple。

        - **attn_out** (Tensor) - 更新后的attention输出，数据类型与输出shape和 `prev_attn_out` 保持一致。
        - **softmax_max** (Tensor) - 更新后的softmax的max，数据类型与输出shape和 `prev_softmax_max` 保持一致。
        - **softmax_sum** (Tensor) - 更新后的softmax的sum，数据类型与输出shape和 `prev_softmax_max` 保持一致。

    异常：
        - **RuntimeError** - 如果 `layout` 为 ``"TND"`` 的时候， `prev_attn_out` 最后一维没有64对齐。
        - **RuntimeError** - 如果 `layout` 为 ``"TND"`` 的时候， `actual_seq_qlen` 没有传入。
        - **RuntimeError** - 如果 `layout` 为 ``"TND"`` 的时候， `actual_seq_qlen` 不是从0到T的非递减序列。
        - **RuntimeError** - 如果 `layout` 为 ``"TND"`` 的时候， `prev_attn_out` shape过大，不满足约束。

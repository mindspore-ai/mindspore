mindspore.ops.speed_fusion_attention
====================================

.. py:function:: mindspore.ops.speed_fusion_attention(query, key, value, head_num, input_layout, *, pse=None, \
                                                      padding_mask=None, atten_mask=None, scale=1.0, keep_prob=1.0, \
                                                      pre_tokens=2147483647, next_tokens=2147483647, inner_precise=0, \
                                                      prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, \
                                                      sparse_mode=0, gen_mask_parallel=True, sync=False, \
                                                      pse_type=1, q_start_idx=None, kv_start_idx=None)

    本接口用于实现self-attention的融合计算。
    当 `pse_type` 为 ``1`` 时，计算公式为：

    .. math::
        attention\_out = Dropout(Softmax(Mask(scale * (pse + query * key^{T}), atten\_mask)), keep\_prob) * value

    当 `pse_type` 为其他有效值时，计算公式为：

    .. math::
        attention\_out = Dropout(Softmax(Mask(scale * (query * key^{T}) + pse, atten\_mask)), keep\_prob) * value

    - B：batch维，取值范围1至2k。
    - S1： `query` 的序列长度，取值范围1至512k。
    - S2： `key` 和 `value` 的序列长度，取值范围1至512k。
    - N1： `query` 的多头数量，取值范围1至256。
    - N2： `key` 和 `value` 的多头数量，且N2必须为N1的因子。
    - D：单头大小，取值范围为16的倍数，最大值为512。
    - H1： `query` 的隐藏层大小，等于N1*D。
    - H2： `key` 和 `value` 的隐藏层大小，等于N2*D。

    .. warning::
        - 这是一个实验性API，可能会发生变更或被删除。
        - 只支持 `Atlas A2` 训练系列产品。

    .. note::
        不支持在 `图模式(mode=mindspore.GRAPH_MODE) <https://www.mindspore.cn/tutorials/zh-CN/master/compile/static_graph.html>`_ 下执行。

    参数：
        - **query** (Tensor) - query Tensor，shape支持： :math:`(B, S1, H1)`、 :math:`(B, N1, S1, D)`、 :math:`(S1, B, H1)`、 :math:`(B, S1, N1, D)` 或 :math:`(T1, N1, D)` 。
        - **key** (Tensor) - key Tensor，shape支持： :math:`(B, S2, H2)`、 :math:`(B, N2, S2, D)`、 :math:`(S2, B, H2)`、 :math:`(B, S2, N2, D)` 或 :math:`(T2, N2, D)` 。
        - **value** (Tensor) - value Tensor，shape支持： :math:`(B, S2, H2)`、 :math:`(B, N2, S2, D)`、 :math:`(S2, B, H2)`、 :math:`(B, S2, N2, D)` 或 :math:`(T2, N2, D)` 。 `value` 的shape应与 `key` 一致。
        - **head_num** (int) - query的多头数量，等于N1。
        - **input_layout** (str) - 指定输入 `query` 、 `key` 和 `value` 的数据格式。值可以为 ``"BSH"`` 、 ``"BNSD"`` 、 ``"SBH"`` 、 ``"BSND"`` 或  ``"TND"`` 。其中 ``"TND"`` 是实验性格式。

          当 `input_layout` 为 ``"TND"`` 时，必须满足以下限制：
          假设有两个列表list_seq_q和list_seq_k表示输入序列的长度。每个列表的值表示batch中每个序列的长度。例如，list_seq_q = [4, 2, 6]，list_seq_k = [10, 3, 9]。
          T1 = sum(list_seq_q) = 12，T2 = sum(list_seq_k) = 22。
          max_seqlen_q = max(list_seq_q)，max_seqlen_k = max(list_seq_k)。
          qk_pointer = sum(list_seq_q * list_seq_k)，表示元素乘积的总和。

          - 两个列表的长度需要相同，列表大小等于batch数量。batch数量最大为1024。
          - 当 `input_layout` 为 ``"TND"`` 时， `actual_seq_qlen` 和 `actual_seq_kvlen` 必须不为 ``None`` 。否则，它们为 ``None`` 。
          - `actual_seq_qlen` 和 `actual_seq_kvlen` 是 `key/value` 序列的累计和，因此必须是非递减的。
          - 如果 `pse` 不为 ``None`` ，则list_seq_q和list_seq_k必须相同。list_seq_q和list_seq_k的最大值必须大于1024。 `pse` 应为 :math:`(B, N1, 1024, S2)` 或 :math:`(1, N1, 1024, S2)`，其中S2等于 max_seqlen_k。
          - `atten_mask` 必须是下三角矩阵，因此 `sparse_mode` 应为2或3，且 `atten_mask` 的shape应为 :math:`(2048, 2048)`。
          - `prefix` 为 ``None`` 。
          - `next_tokens` 为 0， `pre_tokens` 不小于 max_seqlen_q。
          - 当 `sparse_mode` 为3时，每个batch的S1应小于或等于S2。
          - list_seq_k中不应存在0。

    关键字参数：
        - **pse** (Tensor，可选) - 位置编码Tensor，数据类型与 `query` 一致。默认 ``None`` 。
          如果 `S` 大于1024并且使用了下三角的掩码，则仅使用下三角的逆向1024行以进行内存优化。支持的shape为：:math:`(B, N1, S1, S2)` 、:math:`(1, N1, S1, S2)`、:math:`(B, N1, 1024, S2)` 或 :math:`(1, N1, 1024, S2)` 。

          - ALiBi场景： `pse` 必须满足ALiBi规则，并且在下三角场景下 `sparse_mode` 为2或3。在此场景中， `pse` 的shape为 :math:`(B, N1, 1024, S2)` 或 :math:`(1, N1, 1024, S2)`。
          - 非ALiBi场景： `pse` 的shape为 :math:`(B, N1, S1, S2)` 或 :math:`(1, N1, S1, S2)`。
          - `input_layout` 为 ``"TND"`` 时：shape应为 :math:`(B, N1, 1024, S2)` 或 :math:`(1, N1, 1024, S2)`。
          - 如果 `pse_type` 为2或3时，数据类型需为float32，支持的shape为：:math:`(B, N1)` 或 :math:`(N1,)` 。
        - **padding_mask** (Tensor，可选) - 保留参数，尚未使能。默认值为： ``None`` 。
        - **atten_mask** (Tensor，可选) - 注意力掩码Tensor，数据类型为bool、uint8。对于每个元素，0/False表示保留，1/True表示丢弃。shape可以是 :math:`(B, N1, S1, S2)`、:math:`(B, 1, S1, S2)`、:math:`(S1, S2)` 或 :math:`(2048, 2048)`。
          默认值为： ``None`` 。

          - 在压缩场景中，当 `sparse_mode` 为 2、3或4时， `atten_mask` 必须为 :math:`(2048, 2048)`。
          - 当 `sparse_mode` 为5时， `atten_mask` 应该为 :math:`(B, N1, S1, S2)` 或 :math:`(B, 1, S1, S2)`。
          - 当 `sparse_mode` 为0和1时， `atten_mask` 应该是 :math:`(B, N1, S1, S2)`、:math:`(B, 1, S1, S2)` 或 :math:`(S1, S2)`。

        - **scale** (float，可选) - 缩放因子。通常，该值为 1.0 / (D ** 0.5)。默认值为： ``1.0`` 。
        - **keep_prob** (float，可选) - Dropout的保留概率，取值范围为(0.0, 1.0]。默认值为： ``1.0`` 。
        - **pre_tokens** (int，可选) - 用于稀疏计算，表示向前计算的token数量。当 `sparse_mode` 设置为 1、2、3 或5时，此参数无效。默认值为： ``2147483647`` 。
        - **next_tokens** (int，可选) - 稀疏计算的参数，表示向后计算的token数量。当 `sparse_mode` 设置为 1、2、3 或5时，此参数无效。默认值为： ``2147483647`` 。
          `pre_tokens` 的值对应于S1，而 `next_tokens` 的值对应于S2。它们定义了 `atten_mask` 矩阵上的有效区域，并必须确保带宽不为空。以下情况是非法的：

          - pre_tokens < 0 且 next_tokens < 0。
          - (pre_tokens < 0 且 next_tokens >= 0) 且 (next_tokens < abs(pre_tokens) 或 abs(pre_tokens) >= S2)。
          - (pre_tokens >= 0 且 next_tokens < 0) 且 (abs(next_tokens) > pre_tokens 或 abs(next_tokens) >= S1)。

        - **inner_precise** (int，可选) - 保留参数，尚未使能。默认值为： ``0`` 。
        - **prefix** (Union[tuple[int], list[int]]，可选) - 在前缀稀疏计算场景下，每个batch的N值。输入为Tensor时的shape为 :math:`(B,)`，其中B的最大值为32。仅当 `sparse_mode` 为5时，此参数不为 ``None`` 。默认值为： ``None`` 。
          如果 S1 > S2，N 的范围为0到S2。如果S1 <= S2，N的范围为S2 - S1到S2。
        - **actual_seq_qlen** (Union[tuple[int], list[int]]，可选) - 每个batch中query序列的大小，用一个递增值数组表示，最后一个值等于T1。默认值为： ``None`` 。
        - **actual_seq_kvlen** (Union[tuple[int], list[int]]，可选) - 每个batch中key和value序列的大小，用一个递增值数组表示，最后一个值等于T2。默认值为： ``None`` 。
        - **sparse_mode** (int，可选) - 指定稀疏模式。默认值为： ``0``。

          - 0: 表示defaultMask模式。如果未传入 `atten_mask` ，则不执行掩码操作，并忽略 `pre_tokens` 和 `next_tokens` （内部分配为 INT_MAX）。如果传入，需传递完整的 `atten_mask` 矩阵（S1 * S2），表示需计算的部分在 `pre_tokens` 和 `next_tokens` 之间。
          - 1: 表示allMask模式，即传入完整的 `atten_mask` 矩阵。
          - 2: 表示 leftUpCausal 模式，适用于以左顶点划分的下三角场景，需优化的 `atten_mask` 矩阵为 (2048 * 2048)。
          - 3: 表示 rightDownCausal 模式，适用于以右下顶点划分的下三角场景，需优化的 `atten_mask` 矩阵为 (2048 * 2048)。
          - 4: 表示带状场景，即计算 `pre_tokens` 和 `next_tokens` 之间的部分，需优化的 `atten_mask` 矩阵为 (2048 * 2048)。
          - 5: 表示前缀场景，即在 rightDownCausal 的基础上，矩阵左侧添加长度为S1、宽度为N的矩阵。N的值由新输入 `prefix` 获得，每个batch轴的N值不同，目前尚未使能。
          - 6: 表示全局场景，尚未使能。
          - 7: 表示膨胀场景，尚未使能。
          - 8: 表示局部块场景，尚未使能。

        - **gen_mask_parallel** (bool，可选) - Debug参数，控制dropout_gen_mask执行方式的开关，如果为 ``True`` ，dropout_gen_mask并行执行。如果为 ``False`` ，dropout_gen_mask串行执行。尚未使能。默认值为： ``True`` 。
        - **sync** (bool，可选) - Debug参数，控制dropout_gen_mask执行方式的开关，如果为 ``True`` ，dropout_gen_mask同步执行。如果为 ``False`` ，dropout_gen_mask异步执行。尚未使能。默认值为： ``False`` 。
        - **pse_type** (int，可选) - 指定 `pse` 的使用方式。默认值为： ``1`` 。

          - 0: `pse` 由外部传入，计算流程是先mul `scale` 再add `pse` 。
          - 1: `pse` 由外部传入，计算流程是先add `pse` 再mul `scale` 。
          - 2: `pse` 由内部生成，生成标准alibi位置信息。内部生成的alibi矩阵0线与 :math:`query * key^{T}` 的左上角对齐。
          - 3: `pse` 由内部生成，生成的alibi位置信息为标准的基础上再做sqrt开平方。内部生成的alibi矩阵0线与 :math:`query * key^{T}` 的左上角对齐。

        - **q_start_idx** (Union[tuple[int], list[int]]，可选) - 长度为1的序列，默认值为： ``None``。 `pse_type` 配置为2或3时，表示内部生成alibi编码在S1方向偏移的格数，正数表示0对角线向上移动。
        - **kv_start_idx** (Union[tuple[int], list[int]]，可选) - 长度为1的序列，默认值为： ``None``。 `pse_type` 配置为2或3时，表示内部生成alibi编码在S2方向偏移的格数，正数表示0对角线向上移动。

    返回：
        一个Tensor元组，包含 `attention_out`、 `softmax_max`、 `softmax_sum`、 `softmax_out`、 `seed`、 `offset` 和 `numels` 。

        - `attention_out` 是输出Tensor，其shape和dtype与 `query` 相同。
        - `softmax_max` 是Softmax计算的Max中间结果，用于反向计算。
        - `softmax_sum` 是Softmax计算的Sum中间结果，用于反向计算。
        - `softmax_out` 是保留参数。
        - `seed` 是生成的seed，用于Dropout。
        - `offset` 是生成的offset，用于Dropout。
        - `numels` 是生成的dropout_mask的长度。

    异常：
        - **TypeError** - `query`、 `key` 和 `value` 的数据类型不同。
        - **TypeError** - `atten_mask` 的数据类型不是bool或uint8。
        - **TypeError** - `scale` 或 `keep_prob` 不是float类型。
        - **TypeError** - `input_layout` 不是字符串。
        - **TypeError** - `head_num` 不是整数。
        - **TypeError** - `sparse_mode` 不是整数。
        - **TypeError** - `pse` 不是Tensor。
        - **TypeError** - `padding_mask` 不是Tensor。
        - **TypeError** - `atten_mask` 不是Tensor。
        - **TypeError** - `pse_type` 不是整数。
        - **ValueError** - `input_layout` 是字符串但不是合法值。
        - **ValueError** - `sparse_mode` 被指定的值不合法。
        - **ValueError** - `pse_type` 被指定的值不合法。

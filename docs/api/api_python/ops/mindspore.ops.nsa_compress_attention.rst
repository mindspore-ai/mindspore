mindspore.ops.nsa_compress_attention
=====================================

.. py:function:: mindspore.ops.nsa_compress_attention(query, key, value, scale_value, head_num, compress_block_size, compress_stride, select_block_size, select_block_count, *, topk_mask=None, atten_mask=None, actual_seq_qlen, actual_cmp_seq_kvlen, actual_sel_seq_kvlen)

    使用NSA Compress Attention算法进行注意力压缩计算（Ascend）。

    该算子通过以下步骤计算压缩注意力：

    1. 计算注意力分数并应用掩码：

       .. math::

           QK = scale \cdot query \cdot key^T

       应用注意力掩码（如果提供）：

       .. math::

           QK = \text{atten_mask}(QK, atten\_mask)

    2. 计算压缩注意力分数：

       .. math::

           P_{cmp} = \text{Softmax}(QK)

    3. 计算注意力输出：

       .. math::

           \text{attentionOut} = P_{cmp} \cdot value

    4. 计算重要性分数（Importance Score）：

       .. math::

           P_{slc}[j] = \sum_{m=0}^{l'/d-1} \sum_{n=0}^{l/d-1} P_{cmp}[l'/d \cdot j - m - n]

       其中 :math:`l'` 为 select KV 序列长度，:math:`l` 为 compress KV 序列长度，:math:`d` 为 compress_stride。

    5. 聚合多头的重要性分数：

       .. math::

           P_{slc}' = \sum_{h=1}^{H} P_{slc}^h

       其中 :math:`H` 为头数。

    6. 应用 TopK 掩码：

       .. math::

           P_{slc}' = \text{topk_mask}(P_{slc}')

    7. 选择 TopK 索引：

       .. math::

           \text{topkIndices} = \text{topk}(P_{slc}')

    .. note::
        - Ascend 平台下内部布局固定为 "TND"。
        - `actual_seq_qlen` 、 `actual_cmp_seq_kvlen` 、 `actual_sel_seq_kvlen` 采用前缀和模式。
        - `compress_block_size` 必须是16的整数倍，支持范围：16到128。
        - `compress_stride` 必须是16的整数倍，支持范围：16到64。
        - `select_block_size` 必须是16的整数倍，支持范围：16到128。
        - `select_block_count` 支持范围：1到32。
        - `compress_block_size >= compress_stride`
        - `select_block_size >= compress_block_size`
        - `select_block_size % compress_stride == 0`
        - 头维度约束：query和key的头维度 `D1` 必须相同，且 `D1 >= D2` （key的头维度大于等于value的头维度）。
        - 头数约束： `N1 >= N2` （query的头数大于等于key的头数）且 `N1 % N2 == 0` （query的头数必须是key头数的整数倍）。
        - `D1` 和 `D2` 必须是16的整数倍。

    参数：
        - **query** (Tensor) - 查询张量，形状为 `(T1, N1, D1)` ，其中 `T1` 为查询序列长度，`N1` 为查询头数，`D1` 为头维度。dtype类型为 float16 或 bfloat16。必选参数。
        - **key** (Tensor) - 键张量，形状为 `(T2, N2, D1)` ，其中 `T2` 为键序列长度，`N2` 为键头数，`D1` 为头维度（与 `query` 相同）。dtype类型为 float16 或 bfloat16。必选参数。
        - **value** (Tensor) - 值张量，形状为 `(T2, N2, D2)` ，其中 `T2` 为值序列长度，`N2` 为值头数（与 `key` 相同），`D2` 为值头维度。dtype类型为 float16 或 bfloat16。必选参数。
        - **scale_value** (float) - 注意力分数的缩放因子。必选参数。
        - **head_num** (int) - 注意力头数，应等于 `query` 的头数 `N1` 。必选参数。
        - **compress_block_size** (int) - 压缩滑窗大小。必选参数。
        - **compress_stride** (int) - 相邻滑窗间距。必选参数。
        - **select_block_size** (int) - 选择块大小。必选参数。
        - **select_block_count** (int) - 选择块个数。必选参数。

    关键字参数：
        - **topk_mask** (Tensor, 可选) - TopK掩码张量，形状为 `(S1, S2)` ，其中 `S1` 为查询序列长度，`S2` 为select KV序列长度。dtype为 bool。默认值： ``None``。
        - **atten_mask** (Tensor, 可选) - 注意力掩码张量，形状为 `(S1, S2)` ，其中 `S1` 为查询序列长度，`S2` 为compress KV序列长度。dtype为 bool。默认值： ``None``。
        - **actual_seq_qlen** (Union[tuple[int], list[int]]) - 批次query序列长度（前缀和）。
        - **actual_cmp_seq_kvlen** (Union[tuple[int], list[int]]) - 批次compress KV序列长度（前缀和）。
        - **actual_sel_seq_kvlen** (Union[tuple[int], list[int]]) - 批次select KV序列长度（前缀和）。

    返回：
        tuple，包含四个张量。

        - **attention_out** (Tensor) - 注意力输出张量，形状为 `(T1, N1, D2)` 。
        - **topk_indices_out** (Tensor) - TopK索引张量，形状为 `(T1, N2, select_block_count)` ，其中 `T1` 为查询序列长度，`N2` 为键头数。
        - **softmax_max_out** (Tensor) - Softmax计算的Max中间结果，形状为 `(T1, N1, 8)` 。
        - **softmax_sum_out** (Tensor) - Softmax计算的Sum中间结果，形状为 `(T1, N1, 8)` 。

    异常：
        - **TypeError** - 如果输入参数类型不正确。
        - **ValueError** - 如果输入张量形状或参数不满足约束。

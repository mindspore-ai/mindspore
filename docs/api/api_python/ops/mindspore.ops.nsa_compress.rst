mindspore.ops.nsa_compress
==========================

.. py:function:: mindspore.ops.nsa_compress(input, weight, compress_block_size, compress_stride, *, actual_seq_len) -> Tensor

    使用 NSA Compress 算法在 KV 序列维度进行压缩，以降低长上下文训练中的注意力计算开销。

    .. note::
        - 内部布局固定为 ``"TND"``。
        - `actual_seq_len` 采用前缀和模式。必须为非递减整数序列，且末元素等于 ``T``。前缀和示例：若每段长度为 ``[s1, s2, s3]``，则 ``actual_seq_len = (s1, s1 + s2, s1 + s2 + s3)``，其末元素等于 ``T``。
        - 滑窗在每个段内独立进行，不跨段；各段压缩结果按原顺序拼接。
        - ``D`` 必须为 16 的倍数且不大于 256；``1 <= N <= 128``。
        - `compress_block_size` 必须为 16 的倍数且不大于 128；
        - `compress_stride` 必须为 16 的倍数，且 ``16 <= compress_stride <= compress_block_size``。

    参数：
        - **input** (Tensor) - 形状为 ``(T, N, D)``，数据类型为 float16 或 bfloat16。
        - **weight** (Tensor) - 形状为 ``(compress_block_size, N)``，与 `input` 的数据类型一致。
        - **compress_block_size** (int) - 压缩滑窗大小。
        - **compress_stride** (int) - 相邻滑窗间距。

    关键字参数：
        - **actual_seq_len** (Union[tuple[int], list[int]]) - 批次序列长度（前缀和），必须为非递减整数序列，且末元素等于 ``T``。

    返回：
        Tensor。形状为 ``(T', N, D)``，数据类型与 `input` 相同。 :math:`T'` 由 `actual_seq_len`、 `compress_block_size`、 `compress_stride` 联合决定。
        设每段长度为 :math:`L_i` （由前缀和差分得到），则
        :math:`T' = \sum_i \max\big(0,\; 1 + \big\lfloor \frac{L_i - \mathrm{compress\_block\_size}}
        {\mathrm{compress\_stride}} \big\rfloor\big)`。

    异常：
        - **TypeError** - `input` 不是 Tensor。
        - **TypeError** - `weight` 不是 Tensor。
        - **TypeError** - `input` 与 `weight` 的数据类型不一致。
        - **TypeError** - 数据类型不是 float16/bfloat16。
        - **TypeError** - `compress_block_size` 不是 int。
        - **TypeError** - `compress_stride` 不是 int。
        - **TypeError** - `actual_seq_len` 不是由 int 构成的 tuple/list。
        - **RuntimeError** - `input` 的秩不是 3。
        - **RuntimeError** - `weight` 的秩不是 2。
        - **RuntimeError** - ``weight.shape[0] != compress_block_size``。
        - **RuntimeError** - ``weight.shape[1] != N`` （其中 ``N`` 为 `input` 的第 2 维）。
        - **RuntimeError** - ``D % 16 != 0``。
        - **RuntimeError** - ``D > 256``。
        - **RuntimeError** - ``N < 1``。
        - **RuntimeError** - ``N > 128``。
        - **RuntimeError** - `compress_block_size` 不是 16 的倍数。
        - **RuntimeError** - `compress_block_size` 不在 ``[16, 128]``。
        - **RuntimeError** - `compress_stride` 不是 16 的倍数。
        - **RuntimeError** - `compress_stride` 不在 ``[16, compress_block_size]``。
        - **RuntimeError** - `actual_seq_len` 为空。
        - **RuntimeError** - `actual_seq_len` 不是非递减序列。
        - **RuntimeError** - `actual_seq_len` 包含非正元素。
        - **RuntimeError** - `actual_seq_len` 的末元素不等于 ``T``。

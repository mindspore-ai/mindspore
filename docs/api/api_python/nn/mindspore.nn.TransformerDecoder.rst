mindspore.nn.TransformerDecoder
========================================

.. py:class:: mindspore.nn.TransformerDecoder(decoder_layer, num_layers, norm=None)

    Transformer的解码器。多层 `TransformerDecoderLayer` 的堆叠，包括Self Attention层、MultiheadAttention层和FeedForward层。

    参数：
        - **decoder_layer** (Cell) - :class:`mindspore.nn.TransformerDecoderLayer` 的实例。
        - **num_layers** (int) - 解码器层数。
        - **norm** (Cell, 可选) - 层标准化模块。默认值：``None``。

    输入：
        - **tgt** (Tensor) - 目标序列。如果目标序列没有batch，shape是 :math:`(T, E)` ；否则如果 TransformerDecoderLayer中 `batch_first=False` ，则shape为 :math:`(T, N, E)` ，如果 `batch_first=True` ，则shape为 :math:`(T, N, E)`。 :math:`(T)` 是目标序列的长度，:math:`(N)` 是batch个数，:math:`(E)` 是特性个数。数据类型：float16、float32或者float64。
        - **memory** (Tensor) - TransformerEncoder的最后一层输出序列。数据类型：float16、float32或者float64。
        - **tgt_mask** (Tensor, 可选) - 目标序列的掩码矩阵。shape是 :math:`(T, T)` 或 :math:`(N*nhead, T, T)` 。其中 `nhead` 是TransformerDecoderLayer中的 `nhead` 参数。数据类型：：float16、float32、float64或者布尔。默认值：``None``。
        - **memory_mask** (Tensor, 可选) - memory序列的掩码矩阵。shape是 :math:`(T, S)` 。数据类型：：float16、float32、float64或者布尔。默认值：``None``。
        - **tgt_key_padding_mask** (Tensor, 可选) - 目标序列Key矩阵的掩码矩阵。shape是 :math:`(T)` 。数据类型：：float16、float32、float64或者布尔。默认值：``None``。
        - **memory_key_padding_mask** (Tensor, 可选) - memory序列Key矩阵的掩码矩阵。shape是 :math:`(S)` 。数据类型：：float16、float32、float64或者布尔。默认值：``None``。

    输出：
        Tensor。Tensor的shape和dtype与 `tgt` 一致。

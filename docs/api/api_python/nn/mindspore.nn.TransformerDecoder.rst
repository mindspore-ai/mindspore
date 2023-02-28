mindspore.nn.TransformerDecoder
========================================

.. py:class:: mindspore.nn.TransformerDecoder(decoder_layer, num_layers, norm=None)

    Transformer的解码器。多层 `TransformerDecoderLayer` 的堆叠，包括Self Attention层、MultiheadAttention层和FeedForward层。

    参数：
        - **decoder_layer** (Cell) - TransformerDecoderLayer()的实例。
        - **num_layers** (int) - 解码器层数。
        - **norm** (Cell, 可选) - 自定义LayerNorm层。

    输入：
        - **tgt** (Tensor) - 目标序列。
        - **memory** (Tensor) - TransformerEncoder的最后一层输出序列。
        - **tgt_mask** (Tensor, 可选) - 目标序列的掩码矩阵。默认值：``None``。
        - **memory_mask** (Tensor, 可选) - memory序列的掩码矩阵。默认值：``None``。
        - **tgt_key_padding_mask** (Tensor, 可选) - 目标序列Key矩阵的掩码矩阵。默认值：``None``。
        - **memory_key_padding_mask** (Tensor, 可选) - memory序列Key矩阵的掩码矩阵。默认值：``None``。

    输出：
        Tensor。

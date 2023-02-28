mindspore.nn.TransformerEncoder
========================================

.. py:class:: mindspore.nn.TransformerEncoder(encoder_layer, num_layers, norm=None)

    Transformer编码器模块，多层 `TransformerEncoderLayer` 的堆叠，包括MultiheadAttention层和FeedForward层。可以使用此模块构造BERT(https://arxiv.org/abs/1810.04805)模型。

    参数：
        - **encoder_layer** (Cell) - TransformerEncoderLayer()的实例。
        - **num_layers** (int) - 编码器层数。
        - **norm** (Cell, 可选) - 自定义LayerNorm层。

    输入：
        - **src** (Tensor) - 源序列。
        - **src_mask** (Tensor, 可选) - 源序列的掩码矩阵。默认值：``None``。
        - **src_key_padding_mask** (Tensor, 可选) - 源序列Key矩阵的掩码矩阵。默认值：``None``。

    输出：
        Tensor。

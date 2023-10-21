mindspore.nn.TransformerEncoder
========================================

.. py:class:: mindspore.nn.TransformerEncoder(encoder_layer, num_layers, norm=None)

    Transformer编码器模块，多层 `TransformerEncoderLayer` 的堆叠，包括MultiheadAttention层和FeedForward层。可以使用此模块构造BERT(https://arxiv.org/abs/1810.04805)模型。

    参数：
        - **encoder_layer** (Cell) - :class:`mindspore.nn.TransformerEncoderLayer` 的实例。
        - **num_layers** (int) - 编码器层数。
        - **norm** (Cell, 可选) - 自定义LayerNorm层。 默认值： ``None`` 。

    输入：
        - **src** (Tensor) - 源序列。如果源序列没有batch，shape是 :math:`(S, E)` ；否则如果 TransformerEncoderLayer中batch_first=False，则shape为 :math:`(S, N, E)` ，如果batch_first=True，则shape为 :math:`(S, N, E)`。 :math:`(S)` 是源序列的长度, :math:`(N)` 是batch个数， :math:`(E)` 是特性个数。数据类型：float16、float32或者float64。
        - **src_mask** (Tensor, 可选) - 源序列的掩码矩阵。shape是 :math:`(S, S)` 或 :math:`(N*nhead, S, S)` 。其中 `nhead` 是TransformerEncoderLayer中的 `nhead` 参数。数据类型：：float16、float32、float64或者布尔。默认值：``None``。
        - **src_key_padding_mask** (Tensor, 可选) - 源序列Key矩阵的掩码矩阵。shape是 :math:`(S)` 。数据类型：：float16、float32、float64或者布尔。默认值：``None``。

    输出：
        Tensor。Tensor的shape和dtype与 `src` 一致。

    异常：
        - **AssertionError** - 如果 `src_key_padding_mask` 不是布尔或浮点类型。
mindspore.nn.TransformerDecoderLayer
========================================

.. py:class:: mindspore.nn.TransformerDecoderLayer(d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, activation: Union[str, Cell, callable] = 'relu', layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False)

    Transformer的解码器层。Transformer解码器的单层实现，包括Self Attention层、MultiheadAttention层和FeedForward层。

    参数：
        - **d_model** (int) - 输入的特征数。
        - **nhead** (int) - 注意力头的数量。
        - **dim_feedforward** (int) - FeedForward层的维数。默认值：``2048``。
        - **dropout** (float) - 随机丢弃比例。默认值：``0.1``。
        - **activation** (Union[str, callable, Cell]) - 中间层的激活函数，可以输入字符串（``"relu"``、``"gelu"``）、函数接口（``ops.relu``、``ops.gelu``）或激活函数层实例（``nn.ReLU()``、``nn.GELU()``）。默认值：``"relu"``。
        - **layer_norm_eps** (float) - LayerNorm层的eps值，默认值：``1e-5``。
        - **batch_first** (bool) - 如果为 ``True`` 则输入输出Shape为(batch, seq, feature)，反之，Shape为(seq, batch, feature)。默认值： ``False``。
        - **norm_first** (bool) - 如果为 ``True``， 则LayerNorm层位于Self Attention层、MultiheadAttention层和FeedForward层之前，反之，位于其后。默认值： ``False``。

    输入：
        - **tgt** (Tensor) - 目标序列。
        - **memory** (Tensor) - TransformerEncoder的最后一层输出序列。
        - **tgt_mask** (Tensor, 可选) - 目标序列的掩码矩阵。默认值：``None``。
        - **memory_mask** (Tensor, 可选) - memory序列的掩码矩阵。默认值：``None``。
        - **tgt_key_padding_mask** (Tensor, 可选) - 目标序列Key矩阵的掩码矩阵。默认值：``None``。
        - **memory_key_padding_mask** (Tensor, 可选) - memory序列Key矩阵的掩码矩阵∂。默认值：``None``。

    输出：
        Tensor。

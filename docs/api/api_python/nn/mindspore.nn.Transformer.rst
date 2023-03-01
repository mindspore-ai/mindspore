mindspore.nn.Transformer
========================================

.. py:class:: mindspore.nn.Transformer(d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1, activation: Union[str, Cell, callable] = 'relu', custom_encoder: Optional[Cell] = None, custom_decoder: Optional[Cell] = None, layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False)

    Transformer模块，包括编码器和解码器。本模块与原论文的实现不同，原论文在LayerNorm前使用了残差模块。且默认的隐藏层激活函数为 `gelu` 。详情可见 `Attention is all you need <https://arxiv.org/pdf/1706.03762v5.pdf>`_ 。

    参数：
        - **d_model** (int) - Encoder或Decoder输入的特征数。默认值：``512``。
        - **nhead** (int) - 注意力头的数量。默认值：``8``。
        - **num_encoder_layers** (int) - Encoder的层数。默认值：``6``。
        - **num_decoder_layers** (int) - Decoder的层数。默认值：``6``。
        - **dim_feedforward** (int) - FeedForward层的维数。默认值：``2048``。
        - **dropout** (float) - 随机丢弃比例。默认值：``0.1``。
        - **activation** (Union[str, callable, Cell]) - Encoder或Decoder中间层的激活函数，可以输入字符串（``"relu"``、``"gelu"``）、函数接口（``ops.relu``、``ops.gelu``）或激活函数层实例（``nn.ReLU()``、``nn.GELU()``）。默认值：``"relu"``。
        - **custom_encoder** (Cell) - 自定义Encoder层。默认值：``None``。
        - **custom_decoder** (Cell) - 自定义Decoder层。默认值：``None``。
        - **layer_norm_eps** (float) - LayerNorm层的eps值，默认值：``1e-5``。
        - **batch_first** (bool) - 如果为 ``True`` 则输入输出Shape为(batch, seq, feature)，反之，Shape为(seq, batch, feature)。默认值： ``False``。
        - **norm_first** (bool) - 如果为 ``True``，则LayerNorm层位于MultiheadAttention层和FeedForward层之前，反之，位于其后。默认值： ``False``。

    输入：
        - **src** (Tensor) - 源序列。
        - **tgt** (Tensor) - 目标序列。
        - **src_mask** (Tensor, 可选) - 源序列的掩码矩阵。默认值：``None``。
        - **tgt_mask** (Tensor, 可选) - 目标序列的掩码矩阵。默认值：``None``。
        - **memory_mask** (Tensor, 可选) - memory序列的掩码矩阵。默认值：``None``。
        - **src_key_padding_mask** (Tensor, 可选) - 源序列Key矩阵的掩码矩阵。默认值：``None``。
        - **tgt_key_padding_mask** (Tensor, 可选) - 目标序列Key矩阵的掩码矩阵。默认值：``None``。
        - **memory_key_padding_mask** (Tensor, 可选) - memory序列Key矩阵的掩码矩阵。默认值：``None``。

    输出：
        Tensor。

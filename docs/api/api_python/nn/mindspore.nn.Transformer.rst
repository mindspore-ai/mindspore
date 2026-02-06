mindspore.nn.Transformer
========================================

.. py:class:: mindspore.nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', custom_encoder=None, custom_decoder=None, layer_norm_eps=1e-5, batch_first=False, norm_first=False, dtype=mstype.float32)

    Transformer模块，包括编码器和解码器。本模块与原论文的实现不同，原论文在LayerNorm前使用了残差模块。且默认的隐藏层激活函数为 `gelu` 。详情可见 `Attention is all you need <https://arxiv.org/pdf/1706.03762v5.pdf>`_ 。

    参数：
        - **d_model** (int) - Encoder或Decoder输入的特征数。默认值：``512``。
        - **nhead** (int) - MultiheadAttention模块中注意力头的数量。默认值：``8``。
        - **num_encoder_layers** (int) - Encoder的层数。默认值：``6``。
        - **num_decoder_layers** (int) - Decoder的层数。默认值：``6``。
        - **dim_feedforward** (int) - FeedForward层的维数。默认值：``2048``。
        - **dropout** (float) - 随机丢弃比例。默认值：``0.1``。
        - **activation** (Union[str, callable, Cell]) - 中间层的激活函数，可以输入字符串（ ``"relu"`` 、 ``"gelu"`` ）、函数接口（ :func:`mindspore.ops.relu` 、 :func:`mindspore.ops.gelu` ）或激活函数层实例（ :class:`mindspore.nn.ReLU` 、 :class:`mindspore.nn.GELU` ）。默认值： ``'relu'``。
        - **custom_encoder** (Cell) - 自定义Encoder层。默认值：``None``。
        - **custom_decoder** (Cell) - 自定义Decoder层。默认值：``None``。
        - **layer_norm_eps** (float) - LayerNorm层的eps值，默认值：``1e-5``。
        - **batch_first** (bool) - 如果为 ``True`` 则输入输出shape为 :math:`(batch, seq, feature)` ，反之，shape为 :math:`(seq, batch, feature)` 。默认值： ``False``。
        - **norm_first** (bool) - 如果为 ``True``，则LayerNorm层位于MultiheadAttention层和FeedForward层之前，反之，位于其后。默认值： ``False``。
        - **dtype** (:class:`mindspore.dtype`) - Parameter的数据类型。默认值： ``mstype.float32`` 。

    输入：
        - **src** (Tensor) - 源序列。如果源序列没有batch，shape是 :math:`(S, E)` ；否则如果 `batch_first=False` ，则shape为 :math:`(S, N, E)` ，如果 `batch_first=True` ，则shape为 :math:`(N, S, E)`。 :math:`(S)` 是源序列的长度，:math:`(N)` 是batch个数，:math:`(E)` 是特性个数。数据类型：float16、float32或者float64。
        - **tgt** (Tensor) - 目标序列。如果目标序列没有batch，shape是 :math:`(T, E)` ；否则如果 `batch_first=False` ，则shape为 :math:`(T, N, E)` ，如果 `batch_first=True` ，则shape为 :math:`(N, T, E)`。 :math:`(T)` 是目标序列的长度。数据类型：float16、float32或者float64。
        - **src_mask** (Tensor, 可选) - 源序列的掩码矩阵。shape是 :math:`(S, S)` 或 :math:`(N*nhead, S, S)` 。数据类型：float16、float32、float64或者bool。默认值：``None``。
        - **tgt_mask** (Tensor, 可选) - 目标序列的掩码矩阵。shape是 :math:`(T, T)` 或 :math:`(N*nhead, T, T)` 。数据类型：float16、float32、float64或者bool。默认值：``None``。
        - **memory_mask** (Tensor, 可选) - memory序列的掩码矩阵。shape是 :math:`(T, S)` 。数据类型：float16、float32、float64或者bool。默认值：``None``。
        - **src_key_padding_mask** (Tensor, 可选) - 源序列Key矩阵的掩码矩阵。如果目标序列没有batch，shape是 :math:`(S)` ，否则shape为 :math:`(N, S)` 。数据类型：float16、float32、float64或者bool。默认值：``None``。
        - **tgt_key_padding_mask** (Tensor, 可选) - 目标序列Key矩阵的掩码矩阵。如果目标序列没有batch，shape是 :math:`(T)` ，否则shape为 :math:`(N, S)` 。数据类型：float16、float32、float64或者bool。默认值：``None``。
        - **memory_key_padding_mask** (Tensor, 可选) - memory序列Key矩阵的掩码矩阵。如果目标序列没有batch，shape是 :math:`(S)` ，否则shape为 :math:`(N, S)` 。数据类型：float16、float32、float64或者bool。默认值：``None``。

    输出：
        Tensor。如果源序列没有batch，shape是 :math:`(T, E)` ；否则如果 `batch_first=False` ，则shape为 :math:`(T, N, E)` ，如果 `batch_first=True` ，则shape为 :math:`(N, T, E)`。

    异常：
        - **ValueError** - 如果 `src` 和 `tgt` 的batch size不相等。
        - **ValueError** - 如果 `src` 和 `tgt` 的特征数量必须和 `d_model` 的一致。
mindspore.nn.TransformerEncoderLayer
========================================

.. py:class:: mindspore.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', layer_norm_eps=1e-5, batch_first=False, norm_first=False, dtype=mstype.float32)

    Transformer的编码器层。Transformer编码器的单层实现，主要包括Multi-Head Attention、Feed Forward、Add和LayerNorm层。

    TransformerEncoderLayer结构如下图所示：

    .. image:: ../images/TransformerEncoderLayer.png
        :align: center

    参数：
        - **d_model** (int) - 输入的特征数。
        - **nhead** (int) - MultiheadAttention模块中注意力头的数量。
        - **dim_feedforward** (int) - FeedForward层的维数。默认值：``2048``。
        - **dropout** (float) - 随机丢弃比例。默认值：``0.1``。
        - **activation** (Union[str, callable, Cell]) - 中间层的激活函数，可以输入字符串（ ``"relu"`` 、 ``"gelu"`` ）、函数接口（ :func:`mindspore.ops.relu` 、 :func:`mindspore.ops.gelu` ）或激活函数层实例（ :class:`mindspore.nn.ReLU` 、 :class:`mindspore.nn.GELU` ）。默认值： ``'relu'``。
        - **layer_norm_eps** (float) - LayerNorm层的eps值，默认值：``1e-5``。
        - **batch_first** (bool) - 如果为 ``True`` 则输入输出shape为 :math:`(batch, seq, feature)` ，反之，shape为 :math:`(seq, batch, feature)` 。默认值： ``False``。
        - **norm_first** (bool) - 如果为 ``True``， 则LayerNorm层位于MultiheadAttention层和FeedForward层之前，反之，位于其后。默认值： ``False``。
        - **dtype** (:class:`mindspore.dtype`) - Parameter的数据类型。默认值： ``mstype.float32`` 。

    输入：
        - **src** (Tensor) - 源序列。如果源序列没有batch，shape是 :math:`(S, E)` ；否则如果 `batch_first=False` ，则shape为 :math:`(S, N, E)` ，如果 `batch_first=True` ，则shape为 :math:`(N, S, E)`。 :math:`(S)` 是源序列的长度, :math:`(N)` 是batch个数， :math:`(E)` 是特性个数。数据类型：float16、float32或者float64。
        - **src_mask** (Tensor, 可选) - 源序列的掩码矩阵。shape是 :math:`(S, S)` 或 :math:`(N*nhead, S, S)` 。数据类型：float16、float32、float64或者bool。默认值：``None``。
        - **src_key_padding_mask** (Tensor, 可选) - 源序列Key矩阵的掩码矩阵。如果目标序列没有batch，shape是 :math:`(S)` ，否则shape为 :math:`(N, S)` 。数据类型：float16、float32、float64或者bool。默认值：``None``。

    输出：
        Tensor。Tensor的shape和dtype与 `src` 一致。

    异常：
        - **ValueError** - 如果 `activation` 不是str 、 callable 或 Cell的实例。
        - **ValueError** - 如果 `activation` 不是 :class:`mindspore.nn.ReLU` 、 :class:`mindspore.nn.GELU` 的子类、 :func:`mindspore.ops.relu` or :func:`mindspore.ops.gelu`、"relu" 或 "gelu"。
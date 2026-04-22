mindspore.ops.dropout1d
========================

.. py:function:: mindspore.ops.dropout1d(input, p=0.5, training=True)

    在训练期间，以服从伯努利分布的概率 `p` 随机将输入Tensor的某些通道归零（对于形状为 :math:`NCL` 的三维Tensor，其通道特征图指的是后一维 :math:`L` 的一维特征图）。
    例如，在批处理输入中 :math:`i\_th` 批， :math:`j\_th` 通道的 `input[i, j]` `1D` Tensor 是一个待处理数据。
    每个通道将会独立依据伯努利分布概率 `p` 来确定是否被清零。

    论文 `Dropout: A Simple Way to Prevent Neural Networks from Overfitting <http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf>`_ 中提出了该技术，并证明其能有效地减少过度拟合，防止神经元共适应。更多详细信息，请参见 `Improving neural networks by preventing co-adaptation of feature detectors <https://arxiv.org/pdf/1207.0580.pdf>`_ 。

    `dropout1d` 可以提高通道特征映射之间的独立性。

    .. warning::
        从2.9.0（不含）之后版本开始，该接口将新增可选参数 `inplace` ，接口签名将变更为
        ``mindspore.ops.dropout1d(input, p=0.5, training=True, inplace=False)`` 。
        该变更与当前版本兼容，不使用 `inplace` 参数时行为不变。

    参数：
        - **input** (Tensor) - 一个shape为 :math:`(N, C, L)` 或 :math:`(C, L)` 的 `3D` 或 `2D` Tensor，其中N是批处理大小，`C` 是通道数，`L` 是特征长度。其数据类型应为int8、int16、int32、int64、float16、float32或float64。
        - **p** (float，可选) - 通道的丢弃概率，介于 0 和 1 之间，例如 `p` = 0.8，意味着80%的清零概率。默认 ``0.5`` 。
        - **training** (bool，可选) - 是否启用dropout功能。默认 ``True`` 。

    返回：
        Tensor，输出，具有与输入 `input` 相同的shape和数据类型。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `p` 的数据类型不是float。
        - **ValueError** - `p` 值不在 `[0.0, 1.0]` 之间。
        - **ValueError** - `input` 的维度不是 `2D` 或 `3D`。

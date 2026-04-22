mindspore.nn.Dropout2d
======================

.. py:class:: mindspore.nn.Dropout2d(p=0.5)

    在训练期间，服从伯努利分布的概率 `p` 随机将输入Tensor的某些通道置为零（对于shape为 :math:`NCHW` 的四维Tensor，其通道特征图指的是后两维 :math:`HW` 的二维特征图）。
    例如，对于批处理输入中第 :math:`i` 批样本的第 :math:`j` 个通道，是一个待处理的二维Tensor `input[i, j]` 。
    在每次前向传播时，每个通道将会独立地以概率 `p` 来确定是否被置为零。

    `Dropout2d` 可以提高通道特征映射之间的独立性。

    更多参考详见 :func:`mindspore.ops.dropout2d`。

    .. warning::
        2.9.0（不含）之后版本非兼容性变更：前向输入 `x` 将改为 `input`，并新增构造参数 `inplace`。

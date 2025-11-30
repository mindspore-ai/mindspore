mindspore.mint.nn.Dropout2d
===========================

.. py:class:: mindspore.mint.nn.Dropout2d(p=0.5, inplace=False)

    在训练期间，以概率 `p` 随机将输入 tensor 的某些通道置零（对于shape为 :math:`(N, C, H, W)` 的四维 tensor，其通道特征图指的是后两维shape为 :math:`(H, W)` 的二维特征图）。

    例如，在批处理输入中，第 :math:`i` 批的第 :math:`j` 个通道是一个待处理的二维 tensor `input[i, j]`。
    每个通道在每次前向传播时，将会独立地根据伯努利分布的概率 `p` 来确定是否被置零。

    `Dropout2d` 可以提高通道特征映射之间的独立性。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    更多参考详见 :func:`mindspore.mint.nn.functional.dropout2d`。

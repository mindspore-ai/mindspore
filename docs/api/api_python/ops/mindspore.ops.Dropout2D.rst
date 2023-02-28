mindspore.ops.Dropout2D
=======================

.. py:class:: mindspore.ops.Dropout2D(keep_prob=0.5)

    在训练期间，根据概率 :math:`1-keep\_prob` ，随机地将一些通道设置为0，且服从伯努利分布。（对于shape为 :math:`(N, C, H, W)` 的四维Tensor，通道特征图指的是shape为 :math:`(H, W)` 的二维特征图。）

    .. note::
        保持概率 :math:`keep\_prob` 等于 :func:`mindspore.ops.dropout2d` 中的 :math:`1 - p` 。

    更多参考详见 :func:`mindspore.ops.dropout2d`。
    
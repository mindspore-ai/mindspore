mindspore.ops.Dropout3D
=========================

.. py:class:: mindspore.ops.Dropout3D(keep_prob=0.5)

    在训练期间，以服从伯努利分布的概率 :math:`1-keep\_prob` 随机将输入Tensor的某些通道归零。（对于形状为 `NCDHW` 的 `5D` Tensor，其通道特征图指的是后三维 `DHW` 形状的三维特征图）。

    .. note::
        保持概率 :math:`keep\_prob` 等于 :func:`mindspore.ops.dropout3d` 中的 :math:`1 - p` 。

    更多参考详见 :func:`mindspore.ops.dropout3d`。
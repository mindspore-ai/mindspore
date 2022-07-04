mindspore.ops.Dropout3D
========================

.. py:class:: mindspore.ops.Dropout3D(keep_prob=0.5)

    在训练期间，以服从伯努利分布的概率 :math:`1-keep\_prob` 随机将输入Tensor的某些通道归零。（对于形状为 `NCDHW` 的 `5D` Tensor。其通道特征图指的是后两维 `DHW` 形状的三维特征图）。
    例如，在批处理输入中 :math:`i_th` 批， :math:`j_th` 通道的 `input[i, j]` `3D` Tensor 是一个待处理数据。
    每个通道将会独立依据伯努利分布概率 :math:`1-keep\_prob` 来确定是否被清零。

    .. note::
        保持概率 :math:`keep\_prob` 等于 :func:`mindspore.ops.dropout3d` 中的 :math:`1 - p` 。

    更多参考详见 :func:`mindspore.ops.dropout3d`。
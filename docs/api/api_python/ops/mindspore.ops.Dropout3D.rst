mindspore.ops.Dropout3D
=========================

.. py:class:: mindspore.ops.Dropout3D(keep_prob=0.5)

    在训练期间，以服从伯努利分布的概率 :math:`1-keep\_prob` 随机将输入Tensor的某些通道归零。（对于形状为 `NCDHW` 的 `5D` Tensor，其通道特征图指的是后三维 `DHW` 形状的三维特征图）。

    .. note::
        保持概率 :math:`keep\_prob` 等于 :func:`mindspore.ops.dropout3d` 中的 :math:`1 - p` 。

    Dropout3D可以提高feature map之间的独立性。

    参数：
        - **keep_prob** (float) - 输入通道保留率，数值范围在0到1之间，例如 `keep_prob` = 0.8，意味着过滤20%的通道。默认值：0.5。

    输入：
        - **x** (Tensor) - shape为 :math:`(N, C, D, H, W)` 的五维Tensor。N代表批次大小，C代表管道，D代表特征深度，H代表特征高度，W代表特征宽度。数据类型为int8、int16、int32、int64、float16或float32。

    输出：
        - **output** (Tensor) - shape和数据类型与 `x` 相同。
        - **mask** (Tensor) - shape与 `x` 相同，数据类型为bool。

    异常：
        - **TypeError** - `keep_prob` 的数据类型不是float。
        - **ValueError** - `keep_prob` 超出[0.0, 1.0]范围，或者输入的维度不是5。

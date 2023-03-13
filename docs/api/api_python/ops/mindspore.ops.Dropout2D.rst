mindspore.ops.Dropout2D
=======================

.. py:class:: mindspore.ops.Dropout2D(keep_prob=0.5)

    在训练期间，根据概率 :math:`1-keep\_prob` ，随机地将一些通道设置为0，且服从伯努利分布。（对于shape为 :math:`(N, C, H, W)` 的四维Tensor，通道特征图指的是shape为 :math:`(H, W)` 的二维特征图。）

    Dropout2D 可以提高通道特征图之间的独立性。

    .. note::
        保持概率 :math:`keep\_prob` 等于 :func:`mindspore.ops.dropout2d` 中的 :math:`1 - p` 。

    参数：
        - **keep_prob** (float，可选) - 输入通道保留率，数值范围在0到1之间，例如 `keep_prob` = 0.8，意味着过滤20%的通道。默认值：0.5。

    输入：
        - **x** (Tensor) - shape为 :math:`(N, C, H, W)` 的四维张量，其中N是批处理，C是通道数，H是特征高度，W是特征宽度。数据类型应为int8、int16、int32、int64、float16或float32。

    输出：
        - **output** (Tensor) - shape和数据类型与 `x` 相同。
        - **mask** (Tensor) - shape与 `x` 相同，数据类型为bool。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `x` 的数据类型不是int8、int16、int32、int64、float16、float32或float64。
        - **TypeError** - `keep_prob` 的数据类型不是float。
        - **ValueError** - `keep_prob` 值不在 `[0.0，1.0]` 之间。
        - **ValueError** - `x` 的维度不等于4。

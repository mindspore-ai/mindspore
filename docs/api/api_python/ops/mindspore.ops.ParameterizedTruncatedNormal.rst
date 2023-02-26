mindspore.ops.ParameterizedTruncatedNormal
===========================================

.. py:class:: mindspore.ops.ParameterizedTruncatedNormal(seed=0, seed2=0)

    返回一个具有指定shape的Tensor，其数值取自截断正态分布。
    当其shape为 :math:`(batch\_size, *)` 的时候， `mean` 、 `stdevs` 、 `min` 和 `max` 的shape应该为 :math:`()` 或者 :math:`(batch\_size, )` 。

    .. note::
        - 在广播之后，在任何位置， `min` 的值必须严格小于 `max` 的值。
        - 当 `seed` 或 `seed2` 被赋予一个非零值时，该值将被用作种子。否则，将使用一个随机种子。

    参数：
        - **seed** (int，可选) - 随机数种子。默认值：0。
        - **seed2** (int，可选) - 另一个随机种子，避免发生冲突。默认值：0。

    输入：
        - **shape** (Tensor) - 生成Tensor的shape。数据类型必须是int32或者int64。
        - **mean** (Tensor) - 截断正态分布均值。数据类型必须是float16、float32或者float64。
        - **stdevs** (Tensor) - 截断正态分布的标准差。其值必须大于零，数据类型与 `mean` 一致。
        - **min** (Tensor) - 最小截断值，数据类型与 `mean` 一致。
        - **max** (Tensor) - 最大截断值，数据类型与 `mean` 一致。

    输出：
        Tensor，其shape由 `shape` 决定，数据类型与 `mean` 一致。

    异常：
        - **TypeError** - `shape` 、 `mean` 、 `stdevs` 、 `min` 和 `max` 数据类型不支持。
        - **TypeError** - `mean` 、 `stdevs` 、 `min` 和 `max` 的shape不一致。
        - **TypeError** - `shape` 、 `mean` 、 `stdevs` 、 `min` 和 `max` 不全是Tensor。
        - **ValueError** -  当其 `shape` 为 :math:`(batch\_size, *)` 时， `mean` 、 `stdevs` 、 `min` 或者 `max` 的shape不是 :math:`()` 或者 :math:`(batch\_size, )` 。
        - **ValueError** - `shape` 的元素不全大于零。
        - **ValueError** - `stdevs` 的值不全大于零。
        - **ValueError** - `shape` 的的元素个数小于2。
        - **ValueError** - `shape` 不是一维Tensor。

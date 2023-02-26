mindspore.ops.TruncatedNormal
==============================

.. py:class:: mindspore.ops.TruncatedNormal(seed=0, seed2=0, dtype=mstype.float32)

    返回一个具有指定shape的Tensor，其数值取自正态分布。

    生成的值符合正态分布。

    .. note::
        - `shape` 所含元素的值必须大于零。输出长度必须不超过1000000。
        - 当 `seed` 或 `seed2` 被赋予一个非零值时，该值将被用作种子。否则，将使用一个随机种子。

    参数：
        - **seed** (int，可选) - 随机数种子。默认值：0。
        - **seed2** (int，可选) - 另一个随机种子，避免发生冲突。默认值：0。
        - **dtype** (mindspore.dtype，可选) - 指定输出类型。可选值为：mindspore.float16、mindspore.float32和mindspore.float64。默认值：mindspore.float32。

    输入：
        - **shape** (Tensor) - 生成Tensor的shape。数据类型必须是mindspore.int32或者mindspore.int64。

    输出：
        Tensor，其shape由 `shape` 决定，数据类型由 `dtype` 决定。其值在[-2,2]范围内。

    异常：
        - **TypeError** - `shape` 不是Tensor。
        - **TypeError** - `dtype` 或 `shape` 的数据类型不支持。
        - **TypeError** -  `seed` 不是整数。
        - **ValueError** - `shape` 的元素不全大于零。
        - **ValueError** - `shape` 不是一维Tensor。
        - **ValueError** - 输出Tensor的元素个数大于1000000。


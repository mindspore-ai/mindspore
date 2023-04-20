mindspore.ops.gamma
====================

.. py:function:: mindspore.ops.gamma(shape, alpha, beta, seed=None)

    根据伽马分布生成随机数。

    参数：
        - **shape** (tuple) - 指定生成随机数的shape。任意维度的Tensor。
        - **alpha** (Tensor) - :math:`\alpha` 分布的参数。应该大于0且数据类型为float32。
        - **beta** (Tensor) - :math:`\beta` 分布的参数。应该大于0且数据类型为float32。
        - **seed** (int) - 随机数生成器的种子，必须是非负数，默认为 ``None`` ，将视为 ``0`` 。

    返回：
        Tensor。shape是输入 `shape` 、 `alpha` 、 `beta` 广播后的shape。数据类型为float32。

    异常：
        - **TypeError** - `shape` 不是tuple。
        - **TypeError** - `alpha` 或 `beta` 不是Tensor。
        - **TypeError** - `seed` 的数据类型不是int。
        - **TypeError** - `alpha` 或 `beta` 的数据类型不是float32。

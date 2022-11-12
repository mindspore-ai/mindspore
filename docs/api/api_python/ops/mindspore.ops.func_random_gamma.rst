mindspore.ops.random_gamma
==========================

.. py:function:: mindspore.ops.random_gamma(shape, alpha, seed=0, seed2=0)

    根据伽马分布产生成随机数。

    参数：
        - **shape** (Tensor) - 指定生成随机数的shape。任意维度的Tensor。
        - **alpha** (Tensor) - :math:`\alpha` 分布的参数。应该大于0且数据类型为half、float32或者float64。
        - **seed** (int) - 随机数生成器的种子，必须是非负数，默认为0。
        - **seed2** (int) - 随机数生成器的种子，必须是非负数，默认为0。

    返回：
        Tensor。shape是输入 `shape` 、 `alpha` 拼接后的shape。数据类型和alpha一致。

    异常：
        - **TypeError** – `shape` 不是Tensor。
        - **TypeError** – `alpha` 不是Tensor。
        - **TypeError** – `seed` 的数据类型不是int。
        - **TypeError** – `alpha` 的数据类型不是half、float32或者float64。

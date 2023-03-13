mindspore.ops.multinomial
=========================

.. py:function:: mindspore.ops.multinomial(input, num_samples, replacement=True, seed=None)

    根据输入生成一个多项式分布的Tensor。

    .. note::
        输入的行不需要求和为1（当使用值作为权重的情况下），但必须是非负的、有限的，并且和不能为0。

    参数：
        - **input** (Tensor) - 输入的概率值Tensor，必须是一维或二维，数据类型为float32。
        - **num_samples** (int) - 采样的次数。
        - **replacement** (bool, 可选) - 是否是可放回的采样，默认值：True。
        - **seed** (int, 可选) - 随机数种子，用于生成随机数(伪随机数)，必须是非负数。默认值：None。

    返回：
        Tensor，与输入有相同的行数。每行的采样索引数为 `num_samples` 。数据类型为float32。

    异常：
        - **TypeError** - 如果 `input` 不是数据类型不是float32的Tensor。
        - **TypeError** - 如果 `num_samples` 不是int。
        - **TypeError** - 如果 `seed` 既不是int也不是optional。

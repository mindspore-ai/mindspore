mindspore.Tensor.randn_like
============================

.. py:method:: mindspore.Tensor.randn_like(seed=None)

    返回一个与输入大小相同的张量，该张量由均值为0、方差为1的正态分布中的随机数填充。

    参数：
        - **seed** (int, 可选) - 设置随机种子(0到2**32)。

    返回：
        Tensor，形状与self相同。

    异常：
        - **TypeError** - 如果self的数据类型不是int或float。

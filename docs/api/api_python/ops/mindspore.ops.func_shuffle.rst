mindspore.ops.shuffle
=====================

.. py:function:: mindspore.ops.shuffle(x, seed=None)

    沿着Tensor第一维随机打乱数据。

    参数：
        - **x** (Tensor) - 需要随机打乱的Tensor。
        - **seed** (int) - 随机数种子，用于生成随机数，必须为非负数。如果 `seed` 为0，则替换为随机生成的值。默认值是None, 表示使用0作为随机数种子。

    返回：
        Tensor，与输入相同的shape和类型。

    异常：
        - **TypeError** - 如果 `seed` 不是None或非负整数。

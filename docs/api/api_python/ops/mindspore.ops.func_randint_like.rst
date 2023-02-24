mindspore.ops.randint_like
===========================

.. py:function:: mindspore.ops.randint_like(x, low, high, *, dtype=None, seed=None)

    返回一个Tensor，其元素为 [ `low` , `high` ) 区间的随机整数，根据 `x` 决定shape和dtype。

    参数：
        - **x** (Tensor) - 输入的Tensor，用来决定输出Tensor的shape和默认的dtype。
        - **low** (int) - 随机区间的起始值。
        - **high** (int) - 随机区间的结束值。
        - **seed** (int，可选) - 随机种子，必须大于或等于0。默认值：None，值将取0。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`，可选) - 指定的Tensor dtype，必须是int类型的dtype。如果是None，将会使用 `x` 的dtype。默认值：None。

    返回：
        Tensor，shape和dtype被输入指定，其元素为 [ `low` , `high` ) 区间的随机整数。

    异常：
        - **TypeError** - 如果 `seed` 不是非负整数。
        - **TypeError** - 如果 `low` 或 `high` 不是整数。
        - **ValueError** - 如果 `dtype` 不是 `mstype.int_type` 类型。

mindspore.ops.randint
======================

.. py:function:: mindspore.ops.randint(low, high, size, seed=None, *, dtype=None)

    返回一个Tensor，其元素为 [ `low` , `high` ) 区间的随机整数。

    参数：
        - **low** (int) - 随机区间的起始值。
        - **high** (int) - 随机区间的结束值。
        - **size** (tuple) - 新Tensor的shape。
        - **seed** (int，可选) - 随机种子，必须大于或等于0。默认值： ``None`` ，值将取 ``0`` 。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`，可选) - 指定的Tensor dtype，必须是int类型的dtype。如果是 ``None`` ，将会使用 `mindspore.int64` 。默认值： ``None`` 。

    返回：
        Tensor，shape和dtype被输入指定，其元素为 [ `low` , `high` ) 区间的随机整数。

    异常：
        - **TypeError** - 如果 `seed` 不是非负整数。
        - **TypeError** - 如果 `size` 不是tuple。
        - **TypeError** - 如果 `low` 或 `high` 不是整数。
        - **ValueError** - 如果 `dtype` 不是一个 `mstype.int_type` 类型。

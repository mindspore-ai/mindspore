mindspore.mint.randint
======================

.. py:function:: mindspore.mint.randint(low=0, high, size, *, generator=None, dtype=None, device=None) -> Tensor

    返回一个Tensor，shape和dtype由输入决定，其元素为 [ `low` , `high` ) 区间的随机整数。

    参数：
        - **low** (int, 可选) - 随机区间的起始值。默认值：``0`` 。
        - **high** (int) - 随机区间的结束值。
        - **size** (Union[tuple(int), list(int)]) - 新Tensor的shape，如 :math:`(2, 3)`。

    关键字参数：
        - **generator** (:class:`mindspore.Generator`, 可选) - 伪随机数生成器。默认值： ``None`` ，使用默认伪随机数生成器。
        - **dtype** (:class:`mindspore.dtype`，可选) - 指定的Tensor dtype。如果是 ``None`` ，将会使用 `mindspore.int64` 。默认值： ``None`` 。
        - **device** (str, 可选) - 指定Tensor使用的内存来源。仅支持 ``"Ascend"`` 、 ``"npu"``。如果为 ``None`` ，那么将会使用 :func:`mindspore.set_device` 设置的值。默认值： ``None`` 。

    返回：
        Tensor，shape和dtype被输入指定，其元素为 [ `low` , `high` ) 区间的随机整数。

    异常：
        - **TypeError** - 如果 `size` 不是tuple。
        - **TypeError** - 如果 `low` 或 `high` 不是整数。
        - **RuntimeError** - 如果 `device` 是 ``"CPU"`` 。
        - **ValueError** - 如果 `device` 是 ``"GPU"`` 。

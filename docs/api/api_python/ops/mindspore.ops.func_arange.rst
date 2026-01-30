mindspore.ops.arange
=====================

.. py:function:: mindspore.ops.arange(start=0, end=None, step=1, *, dtype=None)

    返回一个在 [ `start`, `end` ) 区间内，步长为 `step` 的tensor。

    参数：
        - **start** (Union[float, int, Tensor], 可选) - 区间的起始值。默认 ``0`` 。
        - **end** (Union[float, int, Tensor], 可选) - 区间的末尾值。默认 ``None`` 时，将 `start` 的值作为区间的末尾值，并将 ``0`` 作为区间的起始值。
        - **step** (Union[float, int, Tensor], 可选) - 值的间隔。默认 ``1`` 。

    关键字参数：
        - **dtype** (mindspore.dtype, 可选) - 指定数据类型。默认 ``None`` 。

    返回：
        Tensor
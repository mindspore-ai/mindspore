mindspore.mint.randint_like
===========================

.. py:function:: mindspore.mint.randint_like(input, low=0, high, *, dtype=None, device=None) -> Tensor

    返回一个Tensor，其元素为 [ `low` , `high` ) 区间的随机整数，根据 `input` 决定shape和dtype。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入的Tensor，用来决定输出Tensor的shape和默认的dtype。
        - **low** (int，可选) - 随机区间的起始值。默认值： ``0`` 。
        - **high** (int) - 随机区间的结束值。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`，可选) - 指定的Tensor dtype。如果是 ``None`` ，将会使用 `input` 的dtype。默认值： ``None`` 。
        - **device** (str, 可选) - 指定Tensor使用的内存来源。仅支持 ``"Ascend"`` 、 ``"npu"``。如果为 ``None`` ，将会使用 `input` 的device。默认值： ``None`` 。

    返回：
        Tensor，shape和dtype被输入指定，其元素为 [ `low` , `high` ) 区间的随机整数。

    异常：
        - **TypeError** - 如果 `low` 或 `high` 不是整数。
        - **RuntimeError** - 如果 `input` 的device是 ``"CPU"`` ，同时 `device` 是 ``None`` 。
        - **RuntimeError** - 如果 `device` 是 ``"CPU"`` 。
        - **ValueError** - 如果 `device` 是 ``"GPU"`` 。

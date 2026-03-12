mindspore.mint.count_nonzero
============================

.. py:function:: mindspore.mint.count_nonzero(input, dim=None)

    在给定维度 `dim` 上，计算 tensor `input` 中非零元素的数量。如果未指定维度，则计算tensor中所有非零元素的数量。

    参数：
        - **input** (Tensor) - 计算非零元素个数的输入。其shape为 :math:`(*)` ，其中 :math:`*` 为任意维度。
        - **dim** (Union[None, int, tuple(int), list(int)], 可选) - 计算非零元素数量的维度。默认值： ``None`` ，表示计算所有非零元素的个数。

    返回：
        Tensor，指定的轴上非零元素数量。

    异常：
        - **TypeError** - `input` 不是Tensor类型。
        - **TypeError** - `dim` 类型不是int、tuple(int)、list(int)或 ``None``。
        - **ValueError** - `dim` 取值不在 :math:`[-input.ndim, input.ndim)` 范围。

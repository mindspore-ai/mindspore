mindspore.ops.nonzero
=====================

.. py:function:: mindspore.ops.nonzero(input, *, as_tuple=False)

    返回所有非零元素下标位置。

    参数：
        - **input** (Tensor) - 输入tensor。

    .. note::
          - Ascend：输入tensor的秩可以等于0，GE后端除外。
          - CPU/GPU：输入tensor秩应大于等于1。

    关键字参数：
        - **as_tuple** (bool, 可选) - 是否以tuple形式输出，默认 ``False`` 。

    返回：
        Tensor或者由tensor组成的tuple。
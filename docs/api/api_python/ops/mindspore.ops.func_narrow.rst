mindspore.ops.narrow
=====================

.. py:function:: mindspore.ops.narrow(input, axis, start, length)

    沿指定轴 `axis` ，从 `start` 位置截取长度为 `length` 的tensor。

    .. warning::
        2.9.0（不含）之后版本中，参数 `axis` 将重命名为 `dim`。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **axis** (int) - 指定轴。
        - **start** (int) - 指定起始位置。
        - **length** (int) - 指定长度。

    返回：
        Tensor

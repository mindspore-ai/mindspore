mindspore.Tensor.as_tensor
==========================

.. py:method:: mindspore.Tensor.as_tensor(data, dtype=None)

    将数据转换为mindspore中的张量。

    参数：
        - **data** (array_like) - 张量的初始数据。可以是列表、元组、NumPy.ndarray、标量和其他类型。
        - **dtype** (mindspore.dtype，可选) - 返回张量的所需数据类型。默认值：如果为"None"，则从数据推断数据类型。

    返回：
        Tensor，数据类型在mindspore的数据类型中。

    平台：
        ``Ascend`` ``GPU`` `` CPU``

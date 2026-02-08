mindspore.Tensor.stride
=======================================

.. py:method:: mindspore.Tensor.stride(dim=None)

    在指定维度 `dim` 中从一个元素跳到下一个元素所必需的步长。当没有参数传入时，返回所有维度的步长的列表。

    参数：
        - **dim** (int，可选) - 指定的维度。默认值： ``None``。

    返回：
        int，返回在指定维度下，从一个元素跳到下一个元素所必需的步长。

    异常：
        - **TypeError** - `dim` 不是int。
mindspore.Tensor.stride
=======================================

.. py:method:: mindspore.Tensor.stride

    在指定维度 `dim中` 从一个元素跳到下一个元素所必需的步长。当没有参数传入时，返回所有维度的步长的列表。

    参数：
        - **dim** (int) - 指定的维度。

    返回：
        Int，返回在指定维度下，从一个元素调到下一个元素所必需的步长。

    异常：
        - **TypeError** - `dim` 不是int。
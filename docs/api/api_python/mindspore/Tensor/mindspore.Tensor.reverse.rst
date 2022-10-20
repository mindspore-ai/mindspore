mindspore.Tensor.reverse
============================

.. py:method:: mindspore.Tensor.reverse(axis)

    按照指定的维度反转tensor

    对输入Tensor按指定维度反转。

    .. warning::
        "axis"的取值范围为[-dims, dims - 1]，"dims"表示"x"的维度长度。

    参数:
        - **axis** (Union[tuple(int), list(int)]): 要反转的维度的索引。

    返回:
            Tensor, 和输入有相同的形状和类型。

    异常：
        - **TypeError** - `axis` 既不是list也不是tuple。
        - **TypeError** - `axis` 的元素不是int。
    
    支持平台:
        ``Ascend`` ``GPU`` ``CPU``

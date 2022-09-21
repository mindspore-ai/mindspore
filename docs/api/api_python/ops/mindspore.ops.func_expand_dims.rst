mindspore.ops.expand_dims
=========================

.. py:function:: mindspore.ops.expand_dims(input_x, axis)

    对输入 `input_x` 在给定的轴上添加额外维度。

    扩展后的Tensor中位置对应 `axis` 的维度为插入的新维度。

    .. note::
        如果指定的 axis 是负数，那么它会从后往前，从1开始计算index。

    参数：
        - **input_x** (Tensor) - 输入Tensor，shape为 :math:`(x_1, x_2, ..., x_R)`。
        - **axis** (int) - 新插入的维度的位置。`axis` 的值必须在范围 `[-input_x.ndim-1, input_x.ndim]` 内。仅接受常量输入。

    返回：
        Tensor，维度在指定轴扩展之后的Tensor，与 `input_x` 的数据类型相同。如果 `axis` 是0，那么它的shape为 :math:`(1, x_1, x_2, ..., x_R)`。

    异常：
        - **TypeError** - 如果 `axis` 不是int。
        - **ValueError** - 如果 `axis` 超出了 :math:`[-a.ndim-1, a.ndim]` 的范围。

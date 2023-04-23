mindspore.ops.padding
=====================

.. py:function:: mindspore.ops.padding(x, pad_dim_size=8)

    通过填充0，将输入Tensor的最后一个维度从1扩展到指定大小。

    参数：
        - **x** (Tensor) - `x` 的shape为 :math:`(x_1, x_2, ..., x_R)`，秩至少为2，它的最后一个维度必须为1。其数据类型为数值型。
        - **pad_dim_size** (int) - 要扩展的 `x` 的最后一个维度的值，该值必须为正数。默认值： ``8`` 。

    返回：
        Tensor，其数据类型和维度必须和输入中的 `x` 保持一致。

    异常：
        - **TypeError** - `pad_dim_size` 的数据类型不是int。
        - **ValueError** - `pad_dim_size` 的值小于1。
        - **ValueError** - `x` 的最后一个维度不等于1。

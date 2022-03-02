mindspore.ops.Padding
=====================

.. py:class:: mindspore.ops.Padding(pad_dim_size=8)

    将输入Tensor的最后一个维度从1扩展到pad_dim_size，其填充值为0。

    **参数：**

    - **pad_dim_size** (int) - 指定填充的大小，待扩展的 `x` 的最后一个维度的值，必须为正数。默认值：8。

    **输入：**

    - **x** (Tensor) - 输入Tensor，其shape为 :math:`(x_1, x_2, ..., x_R)` 。`x` 的秩必须至少为2。 `x` 的最后一个维度必须为1。数据类型为Number。

    **输出：**

    Tensor，其shape是 :math:`(z_1, z_2, ..., z_N)` 。

    **异常：**

    - **TypeError** - `pad_dim_size` 不是int。
    - **ValueError** - `pad_dim_size` 小于1。
    - **ValueError** - `x` 的最后一个维度不等于1。
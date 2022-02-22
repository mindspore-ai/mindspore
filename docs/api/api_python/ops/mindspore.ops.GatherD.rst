mindspore.ops.GatherD
=======================

.. py:class:: mindspore.ops.GatherD

    沿指定轴收集元素。

    对于三维Tensor，输出为：

    .. code-block::

        output[i][j][k] = x[index[i][j][k]][j][k]  # if dim == 0

        output[i][j][k] = x[i][index[i][j][k]][k]  # if dim == 1

        output[i][j][k] = x[i][j][index[i][j][k]]  # if dim == 2

    如果 `x` 是shape为 :math:`(z_0, z_1, ..., z_i, ..., z_{n-1})` ，维度为 `dim` = i的n维Tensor，则 `index` 必须是shape为 :math:`(z_0, z_1, ..., y, ..., z_{n-1})` 的n维Tensor，其中 `y` 大于等于1，输出的shape与 `index` 相同。

    **输入：**

    - **x** (Tensor) - GatherD的输入，任意维度的Tensor。
    - **dim** (int) - 指定索引的轴。数据类型为int32或int64。只能是常量值。
    - **index** (Tensor) - 指定收集元素的索引。支持的数据类型包括：int32，int64。每个索引元素的取值范围为[-x_rank[dim], x_rank[dim])。

    **输出：**

    Tensor，shape为 :math:`(z_1, z_2, ..., z_N)` 的Tensor，数据类型与 `x` 相同。

    **异常：**

    - **TypeError** - `dim` 或 `index` 的数据类型既不是int32，也不是int64。
    - **ValueError** - `x` 的shape长度不等于 `index` 的shape长度。
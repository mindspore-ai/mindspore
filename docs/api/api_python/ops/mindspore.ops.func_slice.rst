mindspore.ops.slice
====================

.. py:function:: mindspore.ops.slice(input_x, begin, size)

    根据指定shape对输入tensor进行切片。

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    .. note::
        `begin` 的起始值为0，`size` 的起始值为1。

    如果 `size[i]` 为-1，则维度i中的所有剩余元素都包含在切片中。这相当于 :math:`size[i] = input\_x.shape(i) - begin[i]` 。

    参数：
        - **input_x** (Tensor) - 输入tensor。
        - **begin** (Union[tuple, list]) - 切片的起始位置，表示每个维度的偏移。
        - **size** (Union[tuple, list]) - 切片的大小。

    返回：
        Tensor

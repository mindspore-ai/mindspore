mindspore.ops.index_add
=======================

.. py:function:: mindspore.ops.index_add(x, indices, y, axis, use_lock=True, check_index_bound=True)

    按照指定轴和索引将输入 `y` 的元素加到输入 `x` 中。

    .. warning::
        2.9.0（不含）之后版本中，参数 `x` 将重命名为 `input`，参数顺序以及 `alpha` 语义会发生变化。

    .. note::
        - `indices` 为一维tensor，并且 :math:`indices.shape[0] = y.shape[axis]` 。
        - `indices` 中元素的取值范围为 :math:`[0, x.shape[axis] - 1]` 。

    参数：
        - **x** (Union[Parameter, Tensor]) - 输入的parameter或tensor。
        - **indices** (Tensor) - 指定索引。
        - **y** (Tensor) - 与 `x` 相加的tensor。
        - **axis** (int) - 指定轴。
        - **use_lock** (bool，可选) - 是否对参数更新过程加锁保护。默认 ``True`` 。
        - **check_index_bound** (bool，可选) - 是否检查 `indices` 边界。默认 ``True`` 。

    返回：
        Tensor

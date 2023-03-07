mindspore.ops.inplace_update
============================

.. py:function:: mindspore.ops.inplace_update(x, v, indices)

    根据 `indices`，将 `x` 中的某些值更新为 `v`。

    .. note::
        `indices` 只能沿着最高轴进行索引。

    参数：
        - **x** (Tensor) - 待更新的Tensor。
        - **v** (Tensor) - 更新的值。
        - **indices** (Union[int, tuple, Tensor]) - 待更新值在原Tensor中的索引。

    返回：
        Tensor，更新后的Tensor。

    异常：
        - **TypeError** - `indices` 不是int或tuple或Tensor。
        - **TypeError** - `indices` 是tuple或Tensor，但是其中的元素不是int。
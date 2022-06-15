mindspore.ops.inplace_sub
=========================

.. py:function:: mindspore.ops.inplace_sub(x, v, indices)

    根据 `indices`，将 `v` 从 `x` 中减去。

    .. note::
        `indices` 只能沿着最高轴进行索引。

    **参数：**

    - **x** (Tensor) - 待更新的Tensor。
    - **v** (Tensor) - 待减去的值。
    - **indices** (Union[int, tuple]) - 待更新值在原Tensor中的索引。

    **返回：**

    Tensor，更新后的Tensor。

    **异常：**

    - **TypeError** - `indices` 不是int或tuple。
    - **TypeError** - `indices` 是元组，但是其中的元素不是int。
    - **ValueError** - `x` 的维度与 `v` 的维度不相等。

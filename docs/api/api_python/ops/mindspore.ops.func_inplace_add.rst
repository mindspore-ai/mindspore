mindspore.ops.inplace_add
=========================

.. py:function:: mindspore.ops.inplace_add(x, v, indices)

    根据 `indices`，将 `x` 中的对应位置加上 `v` 。

    .. note::
        `indices` 只能沿着最高轴进行索引。

    **参数：**

    - **x** (Tensor) - 待更新的Tensor。
    - **v** (Tensor) - 待加上的值。
    - **indices** (Union[int, tuple]) - 待更新值在原Tensor中的索引。

    **返回：**

    Tensor，更新后的Tensor。

    **异常：**

    - **TypeError** - `indices` 不是int或tuple。
    - **TypeError** - `indices` 是元组，但是其中的元素不是int。
    - **ValueError** - `x` 的维度与 `v` 的维度不相等。
    - **ValueError** - `indices` 的长度与 `v.shape[0]` 不相等。
    - **ValueError** - `indices` 的值不属于范围 `[0, x.shape[0])` 。
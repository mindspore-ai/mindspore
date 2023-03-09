mindspore.ops.inplace_update
============================

.. py:function:: mindspore.ops.inplace_update(x, v, indices)

    根据 `indices`，将 `x` 中的某些值更新为 `v`。

    .. note::
        `indices` 只能沿着最高维进行索引。

    参数：
        - **x** (Tensor) - 待更新的Tensor。它可以是以下数据类型之一：float32、float16和int32。
        - **v** (Tensor) - 更新的Tensor，其类型与 `x` 相同，维度大小与 `x` 相同，但第一维度必须与 `indices` 的大小相同。
        - **indices** (Union[int, tuple, Tensor]) - 指定将 `x` 的哪些行更新为 `v` 。可以为int或tuple或Tensor，取值范围[0, `x` 的最高维)。

    返回：
        Tensor，更新后的Tensor。

    异常：
        - **TypeError** - `indices` 不是int或tuple或Tensor。
        - **TypeError** - `indices` 是tuple或Tensor，但是其中的元素不是int。
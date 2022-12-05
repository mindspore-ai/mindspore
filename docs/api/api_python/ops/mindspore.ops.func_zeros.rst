mindspore.ops.zeros
====================

.. py:function:: mindspore.ops.zeros(shape, dtype=None)

    创建一个填满0的Tensor，shape由 `size` 决定， dtype由 `dtype` 决定。

    参数：
        - **shape** (Union[tuple[int], int]) - 用来描述所创建的Tensor的 `shape` 。
        - **dtype** (:class:`mindspore.dtype`) - 用来描述所创建的Tensor的 `dtype`。如果为None，那么将会使用mindspore.float32。默认值：None。

    返回：
        Tensor，dtype和shape由入参决定。

    异常：
        - **TypeError** - 如果 `shape` 既不是int也不是int的元组。

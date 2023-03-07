mindspore.ops.zeros_like
=========================

.. py:function:: mindspore.ops.zeros_like(input, *, dtype=None)

    创建一个填满0的Tensor，shape由 `input` 决定，dtype由 `dtype` 决定。

    参数：
        - **input** (Tensor) - 用来描述所创建的Tensor的shape 。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 用来描述所创建的Tensor的 `dtype`。如果为None，那么将会使用 `input` 的dtype。默认值：None。

    返回：
        Tensor，dtype和shape由入参决定。

    异常：
        - **TypeError** - 如果 `dtype` 不是MindSpore的dtype。

mindspore.ops.ones_like
=======================

.. py:function:: mindspore.ops.ones_like(input, *, dtype=None)

    返回值为1的Tensor，shape和数据类型与输入相同。

    参数：
        - **input** (Tensor) - 任意维度的Tensor。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 用来描述所创建的Tensor的 `dtype`。如果为None，那么将会使用 `input` 的dtype。默认值：None。

    返回：
        Tensor，具有与 `input` 相同的shape并填充了1。

    异常：
        - **TypeError** - `input` 不是Tensor。

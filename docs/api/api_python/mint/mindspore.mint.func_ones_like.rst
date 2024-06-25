mindspore.mint.ones_like
=========================

.. py:function:: mindspore.mint.ones_like(input, *, dtype=None)

    创建一个数值全为1的Tensor，shape和 `input` 相同，dtype由 `dtype` 决定。

    如果 `dtype = None`，输出Tensor的数据类型会和 `input` 一致。

    参数：
        - **input** (Tensor) - 任意维度的Tensor。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 用来描述所创建的Tensor的 `dtype`。如果为 ``None`` ，那么将会使用 `input` 的dtype。默认值： ``None`` 。

    返回：
        Tensor，具有与 `input` 相同的shape并填充了1。

    异常：
        - **TypeError** - `input` 不是Tensor。

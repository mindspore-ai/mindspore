mindspore.mint.zeros_like
==========================

.. py:function:: mindspore.mint.zeros_like(input, *, dtype=None)

    创建一个数值全为0的Tensor，shape和 `input` 相同，dtype由 `dtype` 决定。

    如果 `dtype = None`，输出Tensor的数据类型会和 `input` 一致。

    参数：
        - **input** (Tensor) - 用来描述所创建的Tensor的shape。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 用来描述所创建的Tensor的 `dtype`。如果为 ``None`` ，那么将会使用 `input` 的dtype。默认值： ``None`` 。

    返回：
        返回一个用0填充的Tensor。

    异常：
        - **TypeError** - 如果 `dtype` 不是MindSpore的dtype。

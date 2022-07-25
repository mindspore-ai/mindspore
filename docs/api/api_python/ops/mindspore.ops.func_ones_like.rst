mindspore.ops.ones_like
=======================

.. py:function:: mindspore.ops.ones_like(input_x)

    返回值为1的Tensor，shape和数据类型与输入相同。

    参数：
        - **input_x** (Tensor) - 任意维度的Tensor。

    返回：
        Tensor，具有与 `input_x` 相同的shape和类型，并填充了1。

    异常：
        - **TypeError** - `input_x` 不是Tensor。

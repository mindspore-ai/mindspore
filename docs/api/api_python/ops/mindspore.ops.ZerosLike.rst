mindspore.ops.ZerosLike
=======================

.. py:class:: mindspore.ops.ZerosLike

    返回值为0的Tensor，其shape和数据类型与输入Tensor相同。

    输入：
        - **input_x** (Tensor) - 任意维度的输入Tensor。

    输出：
        Tensor，具有与 `input_x` 相同的shape和数据类型，并填充了0。

    异常：
        - **TypeError** - `input_x` 不是Tensor。

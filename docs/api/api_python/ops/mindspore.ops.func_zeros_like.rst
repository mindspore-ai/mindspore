mindspore.ops.zeros_like
=========================

.. py:function:: mindspore.ops.zeros_like(input_x)

    返回填充值为0的Tensor，其shape和数据类型与 `input_x` 相同。

    参数：
        - **input_x** (Tensor) - 任何维度的输入Tensor。数据类型为int32、int64、float16或float32。

    返回：
        Tensor，具有与 `input_x` 相同的shape和数据类型，但填充了零。

    异常：
        - **TypeError** - 如果 `input_x` 不是Tensor。

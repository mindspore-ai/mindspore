mindspore.ops.ScalarToArray
=============================

.. py:class:: mindspore.ops.ScalarToArray

    将Scalar转换为 `Tensor` 。

    **输入：**

    - **input_x** (Union[int, float]) - ScalarToArray的输入，是Scalar且只能是常量值。

    **输出：**

    Tensor，0维Tensor，其值和输入一致。

    **异常：**

    - **TypeError** - `input_x` 既不是int也不是float。
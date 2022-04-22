mindspore.ops.scalar_to_array
=============================

.. py:function:: mindspore.ops.scalar_to_array(input_x)

    将Scalar转换为 `Tensor` 。

    **参数：**

    - **input_x** (Union[int, float]) - ScalarToArray的输入，是Scalar且只能是常量值。

    **返回：**

    Tensor，0维Tensor，其值和输入一致。

    **异常：**

    - **TypeError** - `input_x` 既不是int也不是float。
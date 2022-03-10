mindspore.ops.ScalarToTensor
==============================

.. py:class:: mindspore.ops.ScalarToTensor

    将Scalar转换为指定数据类型的 `Tensor` 。

    **输入：**

    - **input_x** (Union[int, float]) - 输入是Scalar。只能是常量值。
    - **dtype** (mindspore.dtype) -指定输出的数据类型。只能是常量值。默认值：mindspore.float32。

    **输出：**

    Tensor，0维Tensor，其值和输入一致。

    **异常：**

    - **TypeError** - `input_x` 既不是int也不是float。
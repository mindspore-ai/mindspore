mindspore.ops.scalar_to_tensor
==============================

.. py:function:: mindspore.ops.scalar_to_tensor(input_x, dtype=mstype.float32)

    将标量转换为指定数据类型的tensor。

    参数：
        - **input_x** (Union[bool, int, float]) - 输入标量。只能是常量值。
        - **dtype** (mindspore.dtype) - 返回张量的数据类型。只能是常量值。

    返回：
        Tensor

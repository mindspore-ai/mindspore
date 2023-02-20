mindspore.ops.scalar_to_tensor
==============================

.. py:function:: mindspore.ops.scalar_to_tensor(input_x, dtype=mstype.float32)

    将Scalar转换为指定数据类型的Tensor。

    参数：
        - **input_x** (Union[bool, int, float]) - 输入是Scalar。只能是常量值。
        - **dtype** (mindspore.dtype) - 指定输出的数据类型。只能是常量值。默认值：mindspore.float32。

    返回：
        Tensor，零维Tensor，其值和输入一致。

    异常：
        - **TypeError** - `input_x` 既不是bool, 也不是int，float。
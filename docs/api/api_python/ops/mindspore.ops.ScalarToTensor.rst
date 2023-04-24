mindspore.ops.ScalarToTensor
==============================

.. py:class:: mindspore.ops.ScalarToTensor

    将Scalar转换为指定数据类型的 `Tensor` 。

    更多参考详见 :func:`mindspore.ops.scalar_to_tensor`。

    输入：
        - **input_x** (Union[int, float]) - 输入是Scalar。只能是常量值。
        - **dtype** (mindspore.dtype) - 指定输出的数据类型。只能是常量值。默认值： ``mindspore.float32`` 。

    输出：
        Tensor，零维Tensor，其值和输入一致。

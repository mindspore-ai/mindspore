mindspore.ops.Range
====================

.. py:class:: mindspore.ops.Range(maxlen=1000000)

    产生一系列从 `start` 开始， 步长为 `delta`, 且不超过 `limit` 的数据（不包括 `limit` 本身）。

    三个输入的数据类型必须相同。 函数返回的Tensor的数据类型与输入数据类型保持一致。

    **参数：**

    - **maxlen** (int) - 该算子将会被分配能够存储 `maxlen` 个数据的内存。 该参数是可选的，必须为正数，且默认值为1000000。 如果输出的数量超过 `maxlen` ，将会引起运行时错误。

    **输入：**

    - **start** (Tensor) - 标量Tensor，序列中的第一个数字。数据类型必须为int32或者float32。
    - **limit** (Tensor) - 标量Tensor，序列中的数值上线，不包括其本身。数据类型必须为int32或者float32。
    - **delta** (Tensor) - 标量Tensor，表述序列中数值的步长。数据类型必须为int32或者float32。

    **输出：**

    一维Tensor，数据类型与输入数据类型一致。

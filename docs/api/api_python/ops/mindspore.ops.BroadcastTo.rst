mindspore.ops.BroadcastTo
==========================

.. py:class:: mindspore.ops.BroadcastTo(shape)

    对输入Tensor广播到指定shape。

    将输入shape广播到目标shape。如果目标shape中有-1的维度，它将被该维度中的输入shape的值替换。

    当输入shape广播到目标shape时，它从最后一个维度开始。如果目标shape中有-1的维度，则-1维度不能位于一个不存在的维度中。

    **参数：**

    - **shape** (tuple) - 指定广播的目标shape。

    **输入：**

    - **input_x** (Tensor) - BroadcastTo输入，任意维度的Tensor，数据类型为float16、float32、int32、int8、uint8、bool。

    **输出：**

    Tensor，与目标 `shape` 相同，数据类型与 `input_x` 相同。

    **异常：**

    - **TypeError** - `shape` 不是tuple。
    - **ValueError** - 目标shape和输入shape不兼容，或者目标shape中的-1维度位于一个无效位置。

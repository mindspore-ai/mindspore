mindspore.ops.broadcast_to
==========================

.. py:function:: mindspore.ops.broadcast_to(x, shape)

    将输入shape广播到目标shape。如果目标shape中有-1的维度，它将被该维度中的输入shape的值替换。

    当输入shape广播到目标shape时，它从最后一个维度开始。如果目标shape中有-1维度，则-1维度不能位于一个不存在的维度中。

    参数：
        - **x** (Tensor) - 第一个输入，任意维度的Tensor，数据类型为float16、float32、int32、int8、uint8、bool。
        - **shape** (tuple) - 第二个输入，指定广播到目标 `shape`。

    返回：
        Tensor，shape与目标 `shape` 相同，数据类型与 `x` 相同。

    异常：
        - **TypeError** - `shape` 不是tuple。
        - **ValueError** - 输入shape 无法广播到目标 `shape` ，或者目标 `shape` 中的-1维度位于一个无效位置。

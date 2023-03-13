mindspore.ops.Range
====================

.. py:class:: mindspore.ops.Range(maxlen=1000000)

    返回从 `start` 开始，步长为 `delta` ，且不超过 `limit` （不包括 `limit` ）的序列。

    更多参考详见 :func:`mindspore.ops.range`。

    参数：
        - **maxlen** (int，可选) - 该算子将会被分配能够存储 `maxlen` 个数据的内存。 该参数是可选的，必须为正数，且默认值为1000000。 如果输出的数量超过 `maxlen` ，将会引起运行时错误。

    参数：
        - **start** (Tensor) - 标量Tensor，序列中的第一个数字。数据类型必须为int32，int64，float32或者float64。
        - **limit** (Tensor) - 标量Tensor，序列中的数值上限，不包括其本身。数据类型必须为int32，int64，float32或者float64。
        - **delta** (Tensor) - 标量Tensor，表述序列中数值的步长。数据类型必须为int32，int64，float32或者float64。

    返回：
        一维Tensor，数据类型与输入数据类型一致。

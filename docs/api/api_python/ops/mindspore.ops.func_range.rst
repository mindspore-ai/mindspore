mindspore.ops.range
====================

.. py:function:: mindspore.ops.range(start, end, step, maxlen=1000000)

    返回从 `start` 开始，步长为 `step` ，且不超过 `end` （不包括 `end` ）的序列。

    三个输入必须全为整数或全为浮点数。

    参数：
        - **start** (number) - 序列中的第一个数字。数据类型必须为int32，int64，float32或者float64。
        - **end** (number) - 序列中的数值上限，不包括其本身。数据类型必须为int32，int64，float32或者float64。
        - **step** (number) - 表述序列中数值的步长。数据类型必须为int32，int64，float32或者float64。
        - **maxlen** (int，可选) - 该算子将会被分配能够存储 `maxlen` 个数据的内存。 该参数是可选的，必须为正数，默认值： ``1000000`` 。 如果输出的数量超过 `maxlen` ，将会引起运行时错误。

    返回：
        一维Tensor。若 `start`， `end` ， `step` 全为整数，则输出类型为int64；若 `start`， `end` ， `step` 全为浮点数，则输出类型为float32。

    异常：
        - **TypeError** - `start` ， `end` ， `step` 中既有整数又有浮点数。
        - **TypeError** - `start` ， `end` ， `step` 的数据类型不支持。
        - **ValueError** - `step` 等于0。
        - **ValueError** - `start` 小于等于 `end` ， `step` 小于0。
        - **ValueError** - `start` 大于等于 `end` ， `step` 大于0。

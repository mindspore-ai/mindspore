mindspore.ops.range
====================

.. py:function:: mindspore.ops.range(start, limit, delta)

    返回从 `start` 开始，步长为 `delta` ，且不超过 `limit` （不包括 `limit` ）的序列。

    三个输入必须全为整数或全为浮点数。

    参数：
        - **start** (number) - 序列中的第一个数字。数据类型必须为int32，int64，float32或者float64。
        - **limit** (number) - 序列中的数值上限，不包括其本身。数据类型必须为int32，int64，float32或者float64。
        - **delta** (number) - 表述序列中数值的步长。数据类型必须为int32，int64，float32或者float64。

    返回：
        一维Tensor。若 `start`， `limit` ， `delta` 全为整数，则输出类型为int64；若 `start`， `limit` ， `delta` 全为浮点数，则输出类型为float32。

    异常：
        - **TypeError** - `start` ， `limit` ， `delta` 中既有整数又有浮点数。
        - **TypeError** - `start` ， `limit` ， `delta` 的数据类型不支持。
        - **ValueError** - `delta` 等于0。
        - **ValueError** - `start` 小于等于 `limit` ， `delta` 小于0。
        - **ValueError** - `start` 大于等于 `limit` ， `delta` 大于0。

mindspore.ops.range
====================

.. py:function:: mindspore.ops.range(start, limit, delta)

    返回从 `start` 开始，步长为 `delta` ，且不超过 `limit` （不包括 `limit` ）的序列。

    三个输入的数据类型必须相同。函数返回的Tensor的数据类型与输入数据类型保持一致。

    参数：
        - **start** (Tensor) - 标量Tensor，序列中的第一个数字。数据类型必须为int32，int64，float32或者float64。
        - **limit** (Tensor) - 标量Tensor，序列中的数值上线，不包括其本身。数据类型必须为int32，int64，float32或者float64。
        - **delta** (Tensor) - 标量Tensor，表述序列中数值的步长。数据类型必须为int32，int64，float32或者float64。

    返回：
        一维Tensor，数据类型与输入数据类型一致。

    异常：
        - **TypeError** - `start` ， `limit` ， `delta` 不是标量Tensor。
        - **TypeError** - `start` ， `limit` ， `delta` 的数据类型不一致。
        - **TypeError** - `start` ， `limit` ， `delta` 的数据类型不支持。
        - **ValueError** - `delta` 等于0。
        - **ValueError** - `start` 小于等于 `limit` ， `delta` 小于0。
        - **ValueError** - `start` 大于等于 `limit` ， `delta` 大于0。

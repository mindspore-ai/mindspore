mindspore.ops.dot
==================

.. py:function:: mindspore.ops.dot(x1, x2)

    两个Tensor之间的点积。

    参数：
        - **x1** (Tensor) - 第一个输入的Tensor，数据类型为float16或float32，秩必须大于或等于2。
        - **x2** (Tensor) - 第二个输入的Tensor，数据类型为float16或float32，秩必须大于或等于2。

    返回：
        Tensor， `x1` 和 `x2` 的点积。

    异常：
        - **TypeError** - `x1` 和 `x2` 的数据类型不相同。
        - **TypeError** - `x1` 或 `x2` 的数据类型不是float16或float32。
        - **ValueError** - `x1` 或 `x2` 的秩小于2。
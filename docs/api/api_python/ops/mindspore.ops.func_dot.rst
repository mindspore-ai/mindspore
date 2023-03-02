mindspore.ops.dot
==================

.. py:function:: mindspore.ops.dot(input, other)

    两个Tensor之间的点积。

    参数：
        - **input** (Tensor) - 第一个输入的Tensor，数据类型为float16或float32，秩必须大于或等于2。
        - **other** (Tensor) - 第二个输入的Tensor，数据类型为float16或float32，秩必须大于或等于2。

    返回：
        Tensor， `input` 和 `other` 的点积。

    异常：
        - **TypeError** - `input` 和 `other` 的数据类型不相同。
        - **TypeError** - `input` 或 `other` 的数据类型不是float16或float32。
        - **ValueError** - `input` 或 `other` 的秩小于2。
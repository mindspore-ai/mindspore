mindspore.Tensor.to
===================

.. py:method:: mindspore.Tensor.to(dtype)

    执行Tensor类型的转换。

    参数：
        - **dtype** (dtype.Number) - 输出张量的有效数据类型，只允许常量值。

    返回：
        Tensor，其数据类型为 `dtype`。

    异常：
        - **TypeError** -如果 `dtype` 不是数值类型。

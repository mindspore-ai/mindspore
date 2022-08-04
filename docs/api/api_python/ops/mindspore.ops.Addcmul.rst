mindspore.ops.Addcmul
========================

.. py:class:: mindspore.ops.Addcmul

    执行张量x1与张量x2的逐元素乘积，将结果乘以标量值value，并将其添加到input_data中。

    .. math::
        output[i] = input\_data[i] + value[i] * (x1[i] * x2[i])

    输入：
        - **input_data** (Tensor) - 要添加的张量。
        - **x1** (Tensor) - 要乘以的张量。
        - **x2** (Tensor) - 要乘以的张量。
        - **value** (Tensor) - 张量x1*x2的乘数。

    输出：
        Tensor，具有与x1*x2相同的形状和dtype。

    异常：
        - **TypeError** - 如果 `x1` 、 `x2` 、 `value` 、 `input_data` 的dtype不是张量。
        - **TypeError** - 如果 `input_data` 的dtype不是：float32、float16、int32之一。
        - **TypeError** - 如果 `x1` 或 `x2` 的dtype不是：float32、float16、int32之一.
        - **TypeError** - 如果 `value` 的dtype不是：float32、float16、int32之一。
        - **ValueError** - 如果无法将 `x1` 广播到形状为 `x2` 的张量。
        - **ValueError** - 如果无法将 `value` 广播到形状为 `x1` * `x2` 的张量。
        - **ValueError** - 如果无法将 `input_data` 广播到形状为 `value*(x1*x2)` 的张量。

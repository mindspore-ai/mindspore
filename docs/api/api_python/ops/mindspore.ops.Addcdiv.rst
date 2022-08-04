mindspore.ops.Addcdiv
========================

.. py:class:: mindspore.ops.Addcdiv

    执行张量x1与张量x2的逐元素除法，将结果乘以标量值value，并将其添加到input_data中。

    .. math::
        y[i] = input\_data[i] + value[i] * (x1[i] / x2[i])

    输入：
        - **input_data** (Tensor) - 要添加的张量。
        - **x1** (Tensor) - 分子张量。
        - **x2** (Tensor) - 分母张量。
        - **value** (Tensor) - 张量x1/x2的乘数。

    输出：
        Tensor，具有与x1/x2相同的形状和dtype。

    异常：
        - **TypeError** - 如果 `x1` 、 `x2` 、 `value` 、 `input_data` 的dtype不是张量。
        - **ValueError** - 如果无法将 `x1` 广播到形状为 `x2` 的张量。
        - **ValueError** - 如果无法将 `value` 广播到形状为 `x1/x2` 的张量。
        - **ValueError** - 如果无法将 `input_data` 广播到形状为 `value*(x1/x2)` 的张量。

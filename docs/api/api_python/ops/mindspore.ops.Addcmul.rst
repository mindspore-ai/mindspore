mindspore.ops.Addcmul
========================

.. py:class:: mindspore.ops.Addcmul

    将 `x1` 和 `x2` 的逐元素相乘的结果乘以 `value` ，并将其添加到 `input_data` 中。计算操作如下：

    .. math::
        output[i] = input\_data[i] + value[i] * (x1[i] * x2[i])

    输入：
        - **input_data** (Tensor) - 要添加的Tensor。
        - **x1** (Tensor) - 要乘以的Tensor。
        - **x2** (Tensor) - 要乘以的Tensor。
        - **value** (Tensor) - Tensor x1*x2的乘数。

    输出：
        Tensor，具有与x1*x2相同的shape和dtype。

    异常：
        - **TypeError** - 如果 `x1` 、 `x2` 、 `value` 、 `input_data` 不是Tensor。
        - **TypeError** - 如果 `x1` 、 `x2` 、 `value` 、 `input_data` 的dtype不一致。
        - **ValueError** - 如果无法将 `x1` 广播到 `x2` 。
        - **ValueError** - 如果无法将 `value` 广播到 `x1` * `x2` 。
        - **ValueError** - 如果无法将 `input_data` 广播到 `value*(x1*x2)` 。

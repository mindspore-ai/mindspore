mindspore.ops.csr_softsign
===========================

.. py:function:: mindspore.ops.csr_softsign(x: CSRTensor)

    CSRTensor Softsign激活函数。

    Softsign函数定义为：

    .. math::
        \text{SoftSign}(x) = \frac{x}{1 + |x|}

    参数：
        - **x** (CSRTensor) - CSRTensor，它的数据类型必须为float16或float32。

    返回：
        CSRTensor，数据类型和shape与 `x` 相同。

    异常：
        - **TypeError** - `x` 不是CSRTensor。
        - **TypeError** - `x` 的数据类型既不是float16也不是float32。

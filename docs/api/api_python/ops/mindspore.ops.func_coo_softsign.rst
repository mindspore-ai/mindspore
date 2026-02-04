mindspore.ops.coo_softsign
===========================

.. py:function:: mindspore.ops.coo_softsign(x)

    COOTensor Softsign激活函数。

    Softsign函数定义为：

    .. math::
        \text{SoftSign}(x) = \frac{x}{1 + |x|}

    参数：
        - **x** (COOTensor) - COOTensor的输入。它的数据类型必须为float16或float32。

    返回：
        COOTensor，数据类型和shape与 `x` 相同。

    异常：
        - **TypeError** - `x` 不是COOTensor。
        - **TypeError** - `x` 的数据类型既不是float16也不是float32。

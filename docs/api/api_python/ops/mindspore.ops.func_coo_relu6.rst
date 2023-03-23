mindspore.ops.coo_relu6
========================

.. py:function:: mindspore.ops.coo_relu6(x: COOTensor)

    计算输入COOTensor的ReLU（修正线性单元），其上限为6。

    .. math::
        \text{ReLU6}(x) = \min(\max(0,x), 6)

    返回 :math:`\min(\max(0,x), 6)` 元素的值。

    参数：
        - **x** (COOTensor) - coo_relu6的输入，数据类型为float16或float32。

    返回：
        COOTensor，数据类型和shape与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 的数据类型既不是float16也不是float32。
        - **TypeError** - 如果 `x` 不是COOTensor。

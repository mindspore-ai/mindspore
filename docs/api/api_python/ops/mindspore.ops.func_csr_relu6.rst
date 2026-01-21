mindspore.ops.csr_relu6
========================

.. py:function:: mindspore.ops.csr_relu6(x)

    逐元素计算CSRTensor的ReLU值，其上限为6。

    .. math::
        \text{ReLU6}(x) = \min(\max(0,x), 6)

    返回值为 :math:`\min(\max(0,x), 6)` 。

    参数：
        - **x** (CSRTensor) - csr_relu6的输入，数据类型为float16或float32。

    返回：
        CSRTensor，数据类型和shape与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 的数据类型既不是float16也不是float32。
        - **TypeError** - 如果 `x` 不是CSRTensor。

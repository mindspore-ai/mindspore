mindspore.ops.csr_relu
=======================

.. py:function:: mindspore.ops.csr_relu(x)

    逐元素计算CSRTensor的ReLU（Rectified Linear Unit）激活值。

    返回 max(x, 0) 的值，其中负值神经元将被设置为0，而正值神经元将保持不变。

    .. math::
        ReLU(x) = (x)^+ = \max(0, x)

    参数：
        - **x** (CSRTensor) - csr_relu的输入。

    返回：
        CSRTensor，数据类型和shape与 `x` 相同。

    异常：
        - **TypeError** - `x` 的数据类型不是数值型。
        - **TypeError** - `x` 不是CSRTensor。

mindspore.ops.csr_relu
=======================

.. py:function:: mindspore.ops.csr_relu(x: CSRTensor)

    逐元素计算CSRTensor的ReLU（Rectified Linear Unit）激活值。

    返回 max(x, 0) 的值。负值神经元将被设置为0，正值神经元将保持不变。

    .. math::
        ReLU(x) = (x)^+ = \max(0, x)

    .. note::
        一般来说，与 `ReLUV2` 相比，此算子更常用。且 `ReLUV2` 会多输出一个掩码。

    参数：
        - **x** (CSRTensor) - csr_relu的输入。

    返回：
        CSRTensor，数据类型和shape与 `x` 相同。

    异常：
        - **TypeError** - `x` 的数据类型不是数值型。
        - **TypeError** - `x` 不是CSRTensor。

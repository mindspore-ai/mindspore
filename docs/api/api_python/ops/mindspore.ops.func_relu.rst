mindspore.ops.relu
==================

.. py:function:: mindspore.ops.relu(input)

    对输入Tensor逐元素计算线性修正单元激活函数（Rectified Linear Unit）值。

    返回 :math:`\max(input,\  0)` 的值。负值神经元将被设置为0，正值神经元将保持不变。

    .. math::
        ReLU(input) = (input)^+ = \max(0, input)

    .. note::
        一般来说，与 `ReLUV2` 相比，此算子更常用。且 `ReLUV2` 会多输出一个掩码。

    参数：
        - **input** (Tensor) - 输入Tensor，其数据类型为数值型。

    返回：
        Tensor，其shape和数据类型与输入一致。

    异常：
        - **TypeError** - `input` 的数据类型不是数值型。
        - **TypeError** - `input` 不是Tensor。

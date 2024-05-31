mindspore.mint.nn.functional.relu
=================================

.. py:function:: mindspore.mint.nn.functional.relu(input)

    对输入Tensor逐元素计算线性修正单元激活函数（Rectified Linear Unit）值。

    返回 :math:`\max(input,\  0)` 的值。负值神经元将被设置为0，正值神经元将保持不变。

    .. math::
        ReLU(input) = (input)^+ = \max(0, input)

    ReLU激活函数图：

    .. image:: ../images/ReLU.png
        :align: center

    参数：
        - **input** (Tensor) - 输入Tensor。

    返回：
        Tensor，其shape和数据类型与输入一致。

    异常：
        - **TypeError** - `input` 的数据类型不是数值型。
        - **TypeError** - `input` 不是Tensor。

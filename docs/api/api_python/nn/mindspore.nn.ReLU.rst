mindspore.nn.ReLU
=================

.. py:class:: mindspore.nn.ReLU

    逐元素计算ReLU（Rectified Linear Unit activation function）修正线性单元激活函数。

    .. math::

        \text{ReLU}(input) = (input)^+ = \max(0, input),

    逐元素求 :math:`\max(0, input)` 。
    
    .. note::
        负数输出值会被修改为0，正数输出不受影响。

    ReLU激活函数图：

    .. image:: ../images/ReLU.png
        :align: center

    输入：
        - **input** (Tensor) - 用于计算ReLU的任意维度的Tensor。

    输出：
        Tensor，数据类型和shape与 `input` 相同。

    异常：
        - **TypeError** - `input` 的数据类型不支持。

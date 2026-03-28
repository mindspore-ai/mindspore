mindspore.ops.prelu
===================

.. py:function:: mindspore.ops.prelu(input, weight)

    带参数的线性修正单元激活函数（Parametric Rectified Linear Unit activation function）。

    `Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification <https://arxiv.org/abs/1502.01852>`_ 描述了PReLU激活函数。定义如下：

    .. math::
        prelu(x_i)= \max(0, x_i) + \min(0, w * x_i),

    其中 :math:`x_i` 是输入的一个通道的一个元素，:math:`w` 是通道权重。

    PReLU函数图：

    .. image:: ../images/PReLU2.png
        :align: center

    .. note::
        - 通道数是输入的第二个维度值。当输入的维度小于2时，则没有通道维度，此时通道数等于1。

    参数：
        - **input** (Tensor) - 激活函数的输入Tensor。
        - **weight** (Tensor) - 权重Tensor。其大小为1或者等于输入Tensor `input` 的通道数。

    返回：
        Tensor，其shape和数据类型与 `input` 相同。
        有关详细信息，请参考 :class:`mindspore.mint.nn.PReLU` 。

    异常：
        - **TypeError** - `input` 或 `weight` 不是Tensor。
        - **ValueError** - `weight` 不是0-D或1-D Tensor。

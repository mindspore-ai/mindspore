mindspore.ops.PReLU
===================

.. py:class:: mindspore.ops.PReLU()

    带参数的线性修正单元激活函数（Parametric Rectified Linear Unit activation function）。

    `Delving Deep into Rectifiers:Surpassing Human-Level Performance on ImageNet Classification <https://arxiv.org/abs/1502.01852>`_ 描述了PReLU激活函数。定义如下：

    .. math::
        prelu(x_i)= \max(0, x_i) + \min(0, w * x_i)，

    其中 :math:`x_i` 是输入的一个通道的一个元素，`w` 是通道权重。

    .. note::

        Ascend不支持标量和1维向量的输入x。

    **输入：**

    - **x** (Tensor) - 激活函数的输入Tensor。数据类型为float16或float32。shape为 :math:`(N, C, *)` ，其中 :math:`*` 表示任意的附加维度。
    - **weight** (Tensor) - 权重Tensor。数据类型为float16或float32。weight只可以是向量，长度与输入x的通道数C相同。在GPU设备上，当输入为标量时，shape为1。

    **输出：**

    Tensor，数据类型与 `x` 的相同。

    有关详细信息，请参考 :class:`mindspore.nn.PReLU` 。

    **异常：**

    - **TypeError** - `x` 或  `weight` 的数据类型既不是float16也不是float32。
    - **TypeError** - `x` 或  `weight` 不是Tensor。
    - **ValueError** - `x` 是Ascend上的0-D或1-D Tensor。
    - **ValueError** - `weight` 不是1-D Tensor。

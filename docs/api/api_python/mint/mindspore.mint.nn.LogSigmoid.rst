mindspore.mint.nn.LogSigmoid
=============================

.. py:class:: mindspore.mint.nn.LogSigmoid

    逐元素计算LogSigmoid激活函数。输入可以是任意shape的Tensor。

    LogSigmoid定义为：

    .. math::
        \text{LogSigmoid}(x_{i}) = \log(\frac{1}{1 + \exp(-x_i)}),

    其中，:math:`x_{i}` 是输入Tensor的一个元素。

    LogSigmoid函数图：

    .. image:: ../images/LogSigmoid.png
        :align: center

    输入：
        - **input** (Tensor) - 输入tensor，数据类型为mindspore.bfloat16、mindspore.float16或mindspore.float32。shape为 :math:`(*)` ，其中 :math:`*` 表示任意的附加维度。

    输出：
        Tensor，输出tensor的数据类型和shape与输入相同。



mindspore.mint.nn.functional.logsigmoid
========================================

.. py:function:: mindspore.mint.nn.functional.logsigmoid(input)

    按元素计算LogSigmoid激活函数。输入是任意维度的Tensor。

    LogSigmoid定义为：

    .. math::
        \text{logsigmoid}(x_{i}) = \log(\frac{1}{1 + \exp(-x_{i})})

    其中，:math:`x_{i}` 是输入Tensor的一个元素。

    LogSigmoid函数图：

    .. image:: ../images/LogSigmoid.png
        :align: center

    参数：
        - **input** (Tensor) - logsigmoid的输入，数据类型为bfloat16、float16或float32。shape为 :math:`(*)` ，其中 :math:`*` 表示任意的附加维度。

    返回：
        Tensor，数据类型和shape与 `input` 的相同。

    异常：
        - **TypeError** - `input` 的数据类型既不是bfloat16、float16也不是float32。
        - **TypeError** - `input` 不是一个Tensor。

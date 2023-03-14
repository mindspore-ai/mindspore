mindspore.nn.LogSigmoid
=============================

.. py:class:: mindspore.nn.LogSigmoid

    按元素计算Log Sigmoid激活函数。输入是任意格式的Tensor。

    Log Sigmoid定义为：

    .. math::
        \text{logsigmoid}(x_{i}) = log(\frac{1}{1 + \exp(-x_i)}),

    其中，:math:`x_{i}` 是输入Tensor的一个元素。

    输入：
        - **x** (Tensor) - Log Sigmoid的输入，数据类型为float16或float32。shape为 :math:`(N,*)` ，其中 :math:`*` 表示任意的附加维度。

    输出：
        Tensor，数据类型和shape与 `x` 的相同。

    异常：
        - **TypeError** - `x` 的数据类型既不是float16也不是float32。

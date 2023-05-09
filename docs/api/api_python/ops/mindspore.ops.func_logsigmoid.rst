mindspore.ops.logsigmoid
=============================

.. py:function:: mindspore.ops.logsigmoid(x)

    按元素计算logsigmoid激活函数。输入是任意维度的Tensor。

    logsigmoid定义为：

    .. math::
        \text{logsigmoid}(x_{i}) = \log(\frac{1}{1 + \exp(-x_i)}),

    其中，:math:`x_{i}` 是输入Tensor的一个元素。

    参数：
        - **x** (Tensor) - logsigmoid的输入，数据类型为float16或float32。shape为 :math:`(N,*)` ，其中 :math:`*` 表示任意的附加维度。

    返回：
        Tensor，数据类型和shape与 `x` 的相同。

    异常：
        - **TypeError** - `x` 的数据类型既不是float16也不是float32。

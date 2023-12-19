mindspore.ops.fast_gelu
=========================

.. py:function:: mindspore.ops.fast_gelu(x)

    快速高斯误差线性单元激活函数。

    FastGeLU定义如下：

    .. math::
        \text{output} = \frac {x} {1 + \exp(-1.702 * \left| x \right|)} * \exp(0.851 * (x - \left| x \right|)),

    其中 :math:`x` 是输入元素。

    FastGelu函数图：

    .. image:: ../images/FastGelu.png
        :align: center

    参数：
        - **x** (Tensor) - 计算FastGeLU的输入，数据类型为float16或者float32。

    返回：
        Tensor，其shape和数据类型和 `x` 相同。

    异常：
        - **TypeError** - `x` 数据类型不是float16或者float32。

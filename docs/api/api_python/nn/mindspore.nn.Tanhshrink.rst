mindspore.nn.Tanhshrink
=======================

.. py:class:: mindspore.nn.Tanhshrink

    Tanhshrink激活函数。

    按元素计算Tanhshrink函数，返回一个新的Tensor。

    Tanhshrink函数定义为：

    .. math::
        tanhshrink(x_i) =x_i- \frac{\exp(x_i) - \exp(-x_i)}{\exp(x_i) + \exp(-x_i)} = x_i-\frac{\exp(2x_i) - 1}{\exp(2x_i) + 1},

    其中 :math:`x_i` 是输入Tensor的元素。

    输入：
        - **x** (Tensor) - 任意维度的Tensor。

    输出：
        Tensor，数据类型和shape与 `x` 的相同。

    异常：
        - **TypeError** - `x` 不是一个Tensor。

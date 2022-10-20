mindspore.ops.elu
=================

.. py:function:: mindspore.ops.elu(input_x, alpha=1.0)

    指数线性单元激活函数。

    对输入的每个元素计算ELU。该激活函数定义如下：

    .. math::
            E_{i} =
            \begin{cases}
            x_i, &\text{if } x_i \geq 0; \cr
            \alpha * (\exp(x_i) - 1), &\text{otherwise.}
            \end{cases}

    其中，:math:`x_i` 表示输入的元素，:math:`\alpha` 表示 `alpha` 参数。

    ELU相关图参见 `ELU <https://en.wikipedia.org/wiki/Activation_function#/media/File:Activation_elu.svg>`_  。

    .. math::
        \text{ELU}(x)= \left\{
        \begin{array}{align}
            \alpha(e^{x}  - 1) & \text{if } x \le 0\\
            x & \text{if } x \gt 0\\
        \end{array}\right.

    参数：
        - **input_x** (Tensor) - 用于计算ELU的任意维度的Tensor，数据类型为float16或float32。
        - **alpha** (float) - ELU的alpha值，数据类型为浮点数。默认值：1.0。。

    返回：
        Tensor，输出的shape、数据类型与 `input_x` 相同。

    异常：
        - **TypeError** - 如果 `alpha` 不是float。
        - **TypeError** - 如果 `input_x` 的数据类型既不是float16，也不是float32。
        - **ValueError** - 如果 `alpha` 不等于1.0。

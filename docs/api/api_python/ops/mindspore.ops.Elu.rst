mindspore.ops.Elu
=================

.. py:class:: mindspore.ops.Elu(alpha=1.0)

    指数线性单元激活函数（Exponential Linear Unit activation function）。

    对输入的每个元素计算ELU。该激活函数定义如下：

    .. math::

        \text{ELU}(x)= \left\{
        \begin{array}{align}
            \alpha(e^{x}  - 1) & \text{if } x \le 0\\
            x & \text{if } x \gt 0\\
        \end{array}\right.

    ELU相关图参见 `ELU <https://en.wikipedia.org/wiki/Activation_function#/media/File:Activation_elu.svg>`_  。

    参数：
        - **alpha** (float) - Elu的alpha值，数据类型为浮点数。目前只支持alpha等于1.0，默认值： ``1.0`` 。

    输入：
        - **input_x** (Tensor) - 用于计算Elu的任意维度的Tensor，数据类型为float16,float32或float64。

    输出：
        Tensor，shape和数据类型与 `x` 相同。

    异常：
        - **TypeError** - `alpha` 不是float。
        - **TypeError** - `x` 的数据类型既不是float16，float32也不是float64。
        - **ValueError** - `alpha` 不等于1.0。

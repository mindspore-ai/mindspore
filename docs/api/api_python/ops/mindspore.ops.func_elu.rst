mindspore.ops.elu
==================

.. py:function:: mindspore.ops.elu(input_x, alpha=1.0)

    指数线性单元激活函数。

    对输入的每个元素计算ELU。该激活函数定义如下：

    .. math::

        \text{ELU}(x)= \left\{
        \begin{array}{align}
            \alpha(e^{x}  - 1) & \text{if } x \le 0\\
            x & \text{if } x \gt 0\\
        \end{array}\right.

    其中， :math:`x` 表示输入Tensor `input_x` ， :math:`\alpha` 表示 `alpha` 参数， `alpha` 决定ELU的平滑度。

    ELU函数图：

    .. image:: ../images/ELU.png
        :align: center

    参数：
        - **input_x** (Tensor) - ELU的输入，为任意维度的Tensor，数据类型为float16或float32。
        - **alpha** (float, 可选) - ELU的alpha值，数据类型为float，目前仅支持1.0。默认值： ``1.0`` 。

    返回：
        Tensor，输出的shape和数据类型与 `input_x` 相同。

    异常：
        - **TypeError** - 如果 `alpha` 不是float。
        - **TypeError** - 如果 `input_x` 的数据类型既不是float16，也不是float32。
        - **ValueError** - 如果 `alpha` 不等于1.0。

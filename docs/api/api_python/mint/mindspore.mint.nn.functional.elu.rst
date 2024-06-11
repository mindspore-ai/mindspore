mindspore.mint.nn.functional.elu
===================================

.. py:function:: mindspore.mint.nn.functional.elu(input, alpha=1.0)

    指数线性单元激活函数。

    对输入的每个元素计算ELU。该激活函数定义如下：

    .. math::

        \text{ELU}(x)= \left\{
        \begin{array}{align}
            \alpha(e^{x}  - 1) & \text{if } x \le 0\\
            x & \text{if } x \gt 0\\
        \end{array}\right.

    其中， :math:`x` 表示输入Tensor `input` ， :math:`\alpha` 表示 `alpha` 参数， `alpha` 决定ELU的平滑度。

    ELU函数图：

    .. image:: ../images/ELU.png
        :align: center

    参数：
        - **input** (Tensor) - ELU的输入，为任意维度的Tensor。
        - **alpha** (float, 可选) - ELU的alpha值，数据类型为float。默认值： ``1.0`` 。

    返回：
        Tensor，输出的shape和数据类型与 `input` 相同。

    异常：
        - **TypeError** - 如果 `alpha` 不是float。



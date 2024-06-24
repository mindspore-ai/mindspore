mindspore.nn.ELU
=================

.. py:class:: mindspore.nn.ELU(alpha=1.0)

    指数线性单元激活函数(Exponential Linear Unit activation function)。

    对输入的每个元素计算ELU。该激活函数定义如下：

    .. math::
            E_{i} =
            \begin{cases}
            x_i, &\text{if } x_i \geq 0; \cr
            \alpha * (\exp(x_i) - 1), &\text{otherwise.}
            \end{cases}

    其中，:math:`x_i` 表示输入的元素，:math:`\alpha` 表示 `alpha` 参数。

    ELU函数图：

    .. image:: ../images/ELU.png
        :align: center

    参数：
        - **alpha** (`float`) - ELU的alpha值，数据类型为浮点数，目前仅支持1.0。默认值： ``1.0`` 。

    输入：
        - **input_x** (Tensor) - 用于计算ELU的任意维度的Tensor，数据类型为float16或float32。

    输出：
        Tensor，数据类型和shape与 `input_x` 相同。

    异常：
        - **TypeError** - `alpha` 不是浮点数。
        - **TypeError** - `intpu_x` 的数据类型既不是float16，也不是float32。
        - **ValueError** - `alpha` 不等于1.0。

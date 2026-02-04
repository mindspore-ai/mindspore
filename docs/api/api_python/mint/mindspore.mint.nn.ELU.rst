mindspore.mint.nn.ELU
=====================

.. py:class:: mindspore.mint.nn.ELU(alpha=1.0, inplace=False)

    指数线性单元激活函数(Exponential Linear Unit activation function)。

    对输入的每个元素计算ELU。该激活函数定义如下：

    .. math::
        ELU_{i} =
        \begin{cases}
        x_i, &\text{if } x_i \geq 0; \cr
        \alpha * (\exp(x_i) - 1), &\text{otherwise.}
        \end{cases}

    其中，:math:`x_i` 表示输入的元素，:math:`\alpha` 表示 `alpha` 参数，该参数决定ELU的平滑度。

    ELU函数图：

    .. image:: ../images/ELU.png
        :align: center

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **alpha** (float, 可选) - ELU公式的alpha值。默认值：``1.0``。
        - **inplace** (bool, 可选) - 是否执行in-place操作。默认值：``False``。

    输入：
        - **input** (Tensor) - 任意维度的输入Tensor。

    输出：
        Tensor，输出的shape和数据类型与 `input` 相同。

    异常：
        - **RuntimeError** - 如果 `input` 的数据类型不是float16、float32或bfloat16。
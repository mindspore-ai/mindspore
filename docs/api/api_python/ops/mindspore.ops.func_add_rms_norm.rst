mindspore.ops.add_rms_norm
==========================

.. py:function:: mindspore.ops.add_rms_norm(x1, x2, gamma, epsilon=1e-6)

    AddRmsNorm是一个融合算子，它将RmsNorm和位于其前面的Add算子融合起来，减少了数据搬入搬出的时间。
    其公式如下：

    .. math::
        \begin{array}{ll} \\
            x_i = x1_i + x2_i \\
            y_i=RmsNorm(x_i)=\frac{x_i}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}{ x_i^2}+\varepsilon}}\gamma_i
        \end{array}

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **x1** (Tensor) - AddRmsNorm的输入，支持的数据类型为：float16、float32、bfloat16。
        - **x2** (Tensor) - AddRmsNorm的输入，支持的数据类型为：float16、float32、bfloat16。
        - **gamma** (Tensor) - 可训练参数（:math:`\gamma`），支持的数据类型：float16、float32、bfloat16。
        - **epsilon** (float, 可选) - 一个取值范围为(0, 1]的浮点值，用于避免除零。默认 ``1e-6`` 。

    返回：
        - Tensor，归一化后的结果，shape和数据类型与 `x1` 相同。
        - Tensor，类型为float，表示输入数据标准差的倒数，用于反向梯度计算。
        - Tensor，`x1` 和 `x2` 的和。

    异常：
        - **TypeError** - `x1` 或 `x2` 的数据类型不是float16、float32、bfloat16中的一种。
        - **TypeError** - `gamma` 的数据类型不是float16、float32、bfloat16中的一种。
        - **ValueError** - `epsilon` 不是一个0到1之间的float值。
        - **ValueError** - `gamma` 的秩大于 `x1` 或 `x2` 的秩。
        - **RuntimeError** - `x1` 和 `x2` 的shape不一致。
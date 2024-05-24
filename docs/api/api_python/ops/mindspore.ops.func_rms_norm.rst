mindspore.ops.rms_norm
========================

.. py:function:: mindspore.ops.rms_norm(x, gamma, eps=1e-6)

    RmsNorm(Root Mean Square Layer Normalization), 即均方根标准化。与LayerNorm相比，其保留了缩放不变性，而舍弃了平移不变性。
    其公式如下：

    .. math::
        y = \frac{x_i}{\sqrt{\frac{1}{n}}\sum_{i=1}^{n}{ x_i^2}+\varepsilon  }\gamma_i

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **x** (Tensor) - RmsNorm的输入, 支持的数据类型为: float16, float32, bfloat16。
        - **gamma** (Tensor) - 可训练参数，支持的数据类型: float16, float32, bfloat16。
        - **epsilon** (float, 可选) - 添加到分母中的值（:math:`\epsilon`），以确保数值稳定。默认值： ``1e-6`` 。

    返回：
        - Tensor，归一化后的Tensor，shape和数据类型与 `x` 相同。
        - Tensor, 输入数据标准差的倒数，数据类型和 `x` 相同，用于反向梯度计算。

    异常：
        - **TypeError** - `x` 的数据类型不是float16, float32, bfloat16中的一种。
        - **TypeError** - `gamma` 的数据类型不是float16, float32, bfloat16中的一种。
        - **TypeError** - `x` 和 `gamma` 的数据类型不一致。
        - **ValueError** - `epsilon` 不是一个0到1之间的float值。
        - **ValueError** - `gamma` 的秩大于 `x` 的秩。
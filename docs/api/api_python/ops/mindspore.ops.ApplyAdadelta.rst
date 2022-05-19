mindspore.ops.ApplyAdadelta
============================

.. py:class:: mindspore.ops.ApplyAdadelta

     根据Adadelta算法更新相关参数。
     
     Adadelta算法，具体细节可参考论文 `ADADELTA: AN ADAPTIVE LEARNING RATE METHOD <https://arxiv.org/abs/1212.5701>`_ 。

    .. math::
        \begin{array}{ll} \\
            accum = \rho * accum + (1 - \rho) * grad^2 \\
            \text{update} = \sqrt{\text{accum_update} + \epsilon} * \frac{grad}{\sqrt{accum + \epsilon}} \\
            \text{accum_update} = \rho * \text{accum_update} + (1 - \rho) * update^2 \\
            var -= lr * update
        \end{array}

    其中 :math:`\rho` 代表 `rho` ， :math:`\epsilon` 代表 `epsilon` 。

    `var` 、 `accum` 、 `accum_update` 和 `grad` 的输入遵循隐式类型转换规则，使数据类型一致。如果它们具有不同的数据类型，则较低精度的数据类型将转换为相对最高精度的数据类型。

    **输入：**

    - **var** (Parameter) - 要更新的权重。任意维度，其数据类型为float32或float16。
    - **accum** (Parameter) - :math:`\accum` 要更新的累积，shape和数据类型与 `var` 相同。
    - **accum_update** (Parameter) - 更新公式中的 :math:`\accum_update` ，shape和数据类型与 `var` 相同。
    - **lr** (Union[Number, Tensor]) - :math:`\lr` 学习率，必须是Scalar。数据类型为float32或float16。
    - **rho** (Union[Number, Tensor]) - :math:`\rho` 衰减率，必须是Scalar。数据类型为float32或float16。
    - **epsilon** (Union[Number, Tensor]) - :math:`\epsilon` 加在分母上的值，以确保数值稳定，必须是Scalar。数据类型为float32或float16。
    - **grad** (Tensor) - 梯度，shape和数据类型与 `var` 相同。

    **输出：**

    3个Tensor的元组，更新后的数据。

    - **var** (Tensor) - 与 `var` 相同的shape和数据类型。
    - **accum** (Tensor)- 与 `accum` 相同的shape和数据类型。
    - **accum_update** (Tensor) - 与 `accum_update` 相同的shape和数据类型。

    **异常：**
    
    - **TypeError** - 如果 `var` 、 `accum` 、 `accum_update` 、 `lr` 、 `rho` 、 `epsilon` 或 `grad` 的数据类型既不是float16也不是float32。
    - **TypeError** - 如果 `accum_update` 、 `lr` 、 `rho` 或 `epsilon` 既不是数值型也不是Tensor。
    - **RuntimeError** - 如果`var` 、 `accum` 、 `accum_update` 和 `grad` 不支持数据类型转换。
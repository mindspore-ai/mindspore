mindspore.ops.ApplyAdadelta
============================

.. py:class:: mindspore.ops.ApplyAdadelta

    根据Adadelta算法更新相关参数。

    Adadelta算法，具体细节可参考论文 `ADADELTA: AN ADAPTIVE LEARNING RATE METHOD <https://arxiv.org/abs/1212.5701>`_ 。

    .. math::
        \begin{array}{ll} \\
            \text{accum} = \rho * \text{accum} + (1 - \rho) * \text{grad}^2 \\
            \text{update} = \sqrt{\text{accum_update} +
              \epsilon} * \frac{\text{grad}}{\sqrt{\text{accum} + \epsilon}} \\
            \text{accum_update} = \rho * \text{accum_update} + (1 - \rho) * \text{update}^2 \\
            \text{var} = \text{var} - \text{lr} * \text{update}
        \end{array}

    其中 :math:`\rho` 代表 `rho` ， :math:`\epsilon` 代表 `epsilon` 。

    `var` 、 `accum` 、 `accum_update` 和 `grad` 的输入遵循隐式类型转换规则，使数据类型一致。如果它们具有不同的数据类型，则较低精度的数据类型将转换为相对最高精度的数据类型。

    输入：
        - **var** (Parameter) - 待更新的公式参数 var。数据类型为float32或float16。shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。
        - **accum** (Parameter) - 待更新的公式参数 accum，shape和数据类型与 `var` 相同。
        - **accum_update** (Parameter) - 待更新的公式参数 accum_update，shape和数据类型与 `var` 相同。
        - **lr** (Union[Number, Tensor]) - 学习率，必须是Scalar。数据类型为float32或float16。
        - **rho** (Union[Number, Tensor]) - 衰减率，必须是Scalar。数据类型为float32或float16。
        - **epsilon** (Union[Number, Tensor]) - 加在分母上的值，以确保数值稳定，必须是Scalar。数据类型为float32或float16。
        - **grad** (Tensor) - 梯度，shape和数据类型与 `var` 相同。

    输出：
        3个Tensor的元组，更新后的数据。

        - **var** (Tensor) - 与 `var` 相同的shape和数据类型。
        - **accum** (Tensor)- 与 `accum` 相同的shape和数据类型。
        - **accum_update** (Tensor) - 与 `accum_update` 相同的shape和数据类型。

    异常：
        - **TypeError** - 如果 `var` 、 `accum` 、 `accum_update` 、 `lr` 、 `rho` 、 `epsilon` 或 `grad` 的数据类型既不是float16也不是float32。
        - **TypeError** - 如果 `accum_update` 、 `lr` 、 `rho` 或 `epsilon` 既不是数值型也不是Tensor。
        - **RuntimeError** - 如果 `var` 、 `accum` 、 `accum_update` 和 `grad` 不支持数据类型转换。

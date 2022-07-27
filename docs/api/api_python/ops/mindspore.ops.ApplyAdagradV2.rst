mindspore.ops.ApplyAdagradV2
============================

.. py:class:: mindspore.ops.ApplyAdagradV2(epsilon, update_slots=True)

    根据Adagrad算法更新相关参数。

    Adagrad算法在论文 `Adaptive Subgradient Methods for Online Learning and Stochastic Optimization <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_ 中提出。

    .. math::
        \begin{array}{ll} \\
            accum += grad * grad \\
            var -= lr * grad * \frac{1}{\sqrt{accum} + \epsilon}
        \end{array}

    其中 :math:`\epsilon` 表示 `epsilon` 。

    `var` 、 `accum` 和 `grad` 的输入遵循隐式类型转换规则，使数据类型一致。如果它们具有不同的数据类型，则低精度数据类型将转换为相对最高精度的数据类型。

    .. note::
        `ApplyAdagradV2` 与 `ApplyAdagrad` 不同点在于 `ApplyAdagradV2` 多一个较小的常量值 :math:`\epsilon` 。

    参数：
        - **epsilon** (float) - 添加到分母上的较小值，以确保数值的稳定性。
        - **update_slots** (bool) - 如果为True，则将更新 `accum` 。默认值：True。

    输入：
        - **var** (Parameter) - 要更新的变量。为任意维度，其数据类型为float16或float32。
        - **accum** (Parameter) - 要更新的累积。shape和数据类型必须与 `var` 相同。
        - **lr** (Union[Number, Tensor]) - 学习率，必须是float或具有float16或float32数据类型的Scalar的Tensor。
        - **grad** (Tensor) - 梯度，为一个Tensor。shape和数据类型必须与 `var` 相同。

    输出：
        2个Tensor组成的tuple，更新后的参数。

        - **var** (Tensor) - shape和数据类型与 `var` 相同。
        - **accum** (Tensor) - shape和数据类型与 `accum` 相同。

    异常：
        - **TypeError** - 如果 `var` 、 `accum` 、 `lr` 或 `grad` 的数据类型既不是float16也不是float32。
        - **TypeError** - 如果 `lr` 既不是数值型也不是Tensor。
        - **RuntimeError** - 如果 `var` 、 `accum` 和 `grad` 不支持数据类型转换。

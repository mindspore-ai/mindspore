mindspore.ops.ApplyPowerSign
=============================

.. py:class:: mindspore.ops.ApplyPowerSign

    根据AddSign算法更新相关参数。

    AddSign算法可参阅论文 `Neural Optimizer Search with Reinforcement Learning <https://arxiv.org/abs/1709.07417>`_ 。

    .. math::
        \begin{array}{ll} \\
            m_{t+1} = \beta * m_{t} + (1 - \beta) * g \\
            \text{update} = \exp(\text{logbase} * \text{sign_decay} * sign(g) * sign(m)) * g \\
            var = var - lr_{t+1} * \text{update}
        \end{array}

    :math:`t` 表示更新步数，而 :math:`m` 为一阶矩， :math:`m_{t}` 是上一步的 :math:`m_{t+1}` ， :math:`lr` 表示 `lr` ， :math:`g` 表示 `grad` ， :math:`\beta` 表示 `beta` 。

    所有输入都遵循隐式类型转换规则，以使数据类型一致。如果 `lr` 、 `logbase` 、 `sign_decay` 或 `beta` 是数值型，则会自动转换为Tensor，数据类型与操作中涉及的Tensor的数据类型一致。如果输入是Tensor，并且具有不同的数据类型，则低精度数据类型将转换为最高精度的数据类型。

    .. note::
        目前Ascend平台上暂未开放对float64数据类型的支持。

    输入：
        - **var** (Parameter) - 要更新的变量。数据类型为float64、float32或float16。如果 `var` 的数据类型为float16，则所有输入的数据类型必须与 `var` 相同。shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。
        - **m** (Parameter) - 要更新的变量，shape和数据类型与 `var` 相同。
        - **lr** (Union[Number, Tensor]) - 学习率，应该是Scalar或Tensor，数据类型为float64、float32或float16。
        - **logbase** (Union[Number, Tensor]) - 应该是Scalar或Tensor，数据类型为float64、float32或float16。
        - **sign_decay** (Union[Number, Tensor]) - 应该是Scalar或Tensor，数据类型为float64、float32或float16。
        - **beta** (Union[Number, Tensor]) - 指数衰减率，应该是Scalar或Tensor，数据类型为float64、float32或float16。
        - **grad** (Tensor) - 梯度，shape和数据类型与 `var` 相同。

    输出：
        2个Tensor组成的tuple，更新后的参数。

        - **var** (Tensor) - shape和数据类型与 `var` 相同。
        - **m** (Tensor) - shape和数据类型与 `m` 相同。

    异常：
        - **TypeError** - 如果 `var` 、 `lr` 、 `logbase` 、 `sign_decay` 、 `beta` 或 `grad` 的数据类型不是float16、float32或者float64。
        - **TypeError** - 如果 `lr` 、 `logbase` 、 `sign_decay` 或 `beta` 既不是数值型也不是Tensor。
        - **TypeError** - 如果 `grad` 不是Tensor。
        - **RuntimeError** - 如果 `lr` 、 `logbase` 、 `sign_decay` 和 `grad` 不支持数据类型转换。

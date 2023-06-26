mindspore.ops.ApplyAdaMax
==========================

.. py:class:: mindspore.ops.ApplyAdaMax

    根据AdaMax算法更新相关参数。

    AdaMax优化器是参考 `Adam论文 <https://arxiv.org/abs/1412.6980>`_ 中Adamax优化相关内容所实现的。

    更新公式如下：

    .. math::
        \begin{array}{ll} \\
            m_{t+1} = \beta_1 * m_{t} + (1 - \beta_1) * g \\
            v_{t+1} = \max(\beta_2 * v_{t}, \left| g \right|) \\
            var = var - \frac{l}{1 - \beta_1^{t+1}} * \frac{m_{t+1}}{v_{t+1} + \epsilon}
        \end{array}

    :math:`t` 表示更新步数， :math:`m` 为一阶矩， :math:`m_{t}` 是上一步的 :math:`m_{t+1}` ， :math:`v` 为二阶矩， :math:`v_{t}` 是上一步的 :math:`v_{t+1}` ， :math:`l` 代表学习率 `lr` ， :math:`g` 代表 `grad` ， :math:`\beta_1, \beta_2` 代表 `beta1` 和 `beta2` ， :math:`\beta_1^{t+1}` 代表 `beta1_power` ， :math:`var` 代表要更新的网络参数， :math:`\epsilon` 代表 `epsilon` 。

    `var` 、 `m` 、 `v` 和 `grad` 的输入符合隐式类型转换规则，使数据类型一致。如果它们具有不同的数据类型，则低精度数据类型将转换为相对最高精度的数据类型。

    输入：
        - **var** (Parameter) - 待更新的网络参数，为任意维度。数据类型为float32或float16。其shape为 :math:`(N, *)` ，其中 :math:`*` 为任意数量的额外维度。
        - **m** (Parameter) - 一阶矩，shape与 `var` 相同。数据类型为float32或float16。
        - **v** (Parameter) - 二阶矩。shape与 `var` 相同。数据类型为float32或float16。
        - **beta1_power** (Union[Number, Tensor]) - :math:`beta_1^t` ，必须是Scalar。数据类型为float32或float16。
        - **lr** (Union[Number, Tensor]) - 学习率，公式中的 :math:`l` ，必须是Scalar。数据类型为float32或float16。
        - **beta1** (Union[Number, Tensor]) - 一阶矩的指数衰减率，必须是Scalar。数据类型为float32或float16。
        - **beta2** (Union[Number, Tensor]) - 二阶矩的指数衰减率，必须是Scalar。数据类型为float32或float16。
        - **epsilon** (Union[Number, Tensor]) - 加在分母上的值，以确保数值稳定，必须是Scalar。数据类型为float32或float16。
        - **grad** (Tensor) - 为梯度，是一个Tensor，shape与 `var` 相同。数据类型为float32或float16。

    输出：
        3个Tensor组成的tuple，更新后的数据。

        - **var** (Tensor) - shape和数据类型与 `var` 相同。
        - **m** (Tensor) - shape和数据类型与 `m` 相同。
        - **v** (Tensor) - shape和数据类型与 `v` 相同。

    异常：
        - **TypeError** - 如果 `var` 、 `m` 、 `v` 、 `beta_power` 、 `lr` 、 `beta1` 、 `beta2` 、 `epsilon` 或 `grad` 的数据类型既不是float16也不是float32。
        - **TypeError** - 如果 `beta_power` 、 `lr` 、 `beta1` 、 `beta2` 或 `epsilon` 既不是数值型也不是Tensor。
        - **TypeError** - 如果 `grad` 不是Tensor。
        - **TypeError** - 如果 `var` 、 `m` 、 `v` 和 `grad` 不支持数据类型转换。

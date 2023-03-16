mindspore.ops.ApplyCenteredRMSProp
====================================

.. py:class:: mindspore.ops.ApplyCenteredRMSProp(use_locking=False)

    居中RMSProp算法优化器。

    请参考源代码中的用法： :class:`mindspore.nn.RMSProp` 。

    更新公式如下：

    .. math::
        \begin{array}{ll} \\
            g_{t+1} = \rho g_{t} + (1 - \rho)\nabla Q_{i}(w) \\
            s_{t+1} = \rho s_{t} + (1 - \rho)(\nabla Q_{i}(w))^2 \\
            m_{t+1} = \beta m_{t} + \frac{\eta} {\sqrt{s_{t+1} - g_{t+1}^2 + \epsilon}} \nabla Q_{i}(w) \\
            w = w - m_{t+1}
        \end{array}

    其中 :math:`w` 代表 `var` ， 将会更新。
    :math:`g_{t+1}` 代表 `mean_gradient` ， :math:`g_{t}` 是上一步的 :math:`g_{t+1}` 。
    :math:`s_{t+1}` 代表 `mean_square` ， :math:`s_{t}` 是上一步的 :math:`s_{t+1}` ，
    :math:`m_{t+1}` 代表 `moment` ， :math:`m_{t}` 是上一步的 :math:`m_{t+1}` 。
    :math:`\rho` 代表 `decay` 。 :math:`\beta` 是动量，代表 `momentum` 。
    :math:`\epsilon` 是一个添加在分母上的较小值，以避免被零除，表示 `epsilon` 。
    :math:`\eta` 代表 `learning_rate` 。 :math:`\nabla Q_{i}(w)` 代表 `grad` 。

    .. note::
        `ApplyCenteredRMSProp` 和 `ApplyRMSProp` 的区别在于前者使用居中RMSProp算法，而居中RMSProp算法使用居中第二矩阵的估计（即，归一化的方差），而不是使用（不确定的）第二矩阵的正则RMSProp。这通常有助于训练，但在计算和内存方面消耗更大。

    .. warning::
        在此算法的密集实现中， `mean_gradient` 、 `mean_square` 和 `moment` 在 `grad` 为零时仍将被更新。但在稀疏实现中， `mean_gradient` 、 `mean_square` 和 `moment` 不会在 `grad` 为零的迭代中被更新。

    参数：
        - **use_locking** (bool) - 是否对参数更新增加锁保护。默认值：False。

    输入：
        - **var** (Parameter) - 要更新的权重。
        - **mean_gradient** (Tensor) - 均值梯度，数据类型必须与 `var` 相同。
        - **mean_square** (Tensor) - 均方梯度，数据类型必须与 `var` 相同。
        - **moment** (Tensor) - `var` 的增量，数据类型必须与 `var` 相同。
        - **grad** (Tensor) - 梯度，数据类型必须与 `var` 相同。
        - **learning_rate** (Union[Number, Tensor]) - 学习率。必须是float或Scalar的Tensor，数据类型为float16或float32。
        - **decay** (float) - 衰减率。
        - **momentum** (float) - 动量。
        - **epsilon** (float) - 添加在分母上的较小值，以避免被零除。

    输出：
        Tensor，更新后的数据。

    异常：
        - **TypeError** - 如果 `use_locking` 不是bool。
        - **TypeError** - 如果 `var` 、 `mean_gradient` 、 `mean_square` 、 `moment` 或 `grad` 不是Tensor。
        - **TypeError** - 如果 `learing_rate` 既不是数值型也不是Tensor。
        - **TypeError** - 如果 `learing_rate` 的数据类型既不是float16也不是float32。
        - **TypeError** - 如果 `decay` 、 `momentum` 或 `epsilon` 不是float。

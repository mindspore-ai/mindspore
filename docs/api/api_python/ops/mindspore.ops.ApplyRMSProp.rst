mindspore.ops.ApplyRMSProp
==========================

.. py:class:: mindspore.ops.ApplyRMSProp(use_locking=False)

    实现均方根传播Root Mean Square prop(RMSProp)算法的优化器。请参考源码中的用法 :class:`mindspore.nn.RMSProp` 。

    ApplyRMSProp算法的更新公式如下所示：

    .. math::
        \begin{array}{ll} \\
            s_{t+1} = \rho s_{t} + (1 - \rho)(\nabla Q_{i}(w))^2 \\
            m_{t+1} = \beta m_{t} + \frac{\eta} {\sqrt{s_{t+1} + \epsilon}} \nabla Q_{i}(w) \\
            w = w - m_{t+1}
        \end{array}

    其中 :math:`w` 代表待更新的网络参数 `var` 。
    :math:`s_{t+1}` 为均方梯度 `mean_square` ，:math:`s_{t}` 为上一步的 :math:`s_{t+1}` ，
    :math:`m_{t+1}` 为 `moment` ， :math:`m_{t}` 为上一步的 :math:`m_{t+1}` 。
    :math:`\rho` 为 `decay` 。 :math:`\beta` 为动量项 `momentum` 。
    :math:`\epsilon` 是避免零为除数的平滑项 `epsilon` 。
    :math:`\eta` 为 `learning_rate` 。 :math:`\nabla Q_{i}(w)` 代表 `grad` 。

    .. warning::
        在该算法的稠密实现版本中，"mean_square"和"momemt"即使"grad"为零将仍被更新。但在该稀疏实现版本中，在"grad"为零的迭代"mean_squre"和"moment"将不被更新。

    参数：
        - **use_locking** (bool) - 是否对参数更新加锁保护。默认值: False。

    输入：
        - **var** (Parameter) - 待更新的网络参数。
        - **mean_square** (Tensor) - 均方梯度，数据类型需与 `var` 相同。
        - **moment** (Tensor) - 一阶矩，数据类型需与 `var` 相同。
        - **learning_rate** (Union[Number, Tensor]) - 学习率。需为浮点数或者数据类型为float16或float32的标量矩阵。
        - **grad** (Tensor) - 梯度，数据类型需与 `var` 相同。
        - **decay** (float) - 衰减率。需为常量。
        - **momentum** (float) - 移动平均的动量项momentum。需为常量。
        - **epsilon** (float) - 避免除数为零的平滑项。需为常量。

    输出：
        Tensor，待更新的网络参数。

    异常：
        - **TypeError** - `use_locking` 不是bool类型。
        - **TypeError** - `var` 、 `mean_square` 、 `moment` 或 `decay` 不是Tensor。
        - **TypeError** - `learning_rate` 不是数值也不是Tensor。
        - **TypeError** - `decay` 、 `momentum` 或 `epsilon` 的数据类型非float。
        - **TypeError** - `learning_rate` 的数据类型不是float16或float32。
        - **ValueError** - `decay` 、 `momentum` 或 `epsilon` 不是常量。

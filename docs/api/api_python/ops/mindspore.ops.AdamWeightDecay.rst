mindspore.ops.AdamWeightDecay
=============================

.. py:class:: mindspore.ops.AdamWeightDecay(use_locking=False)

    通过具有权重衰减的自适应矩估计算法（AdamWeightDecay）更新梯度。
    Adam算法在 `Adam：随机优化方法 <https://arxiv.org/abs/1412.6980>`_ 中提出。

    AdamWeightDecay是Adam算法的变体，在 `解耦权重衰变正则化 <https://arxiv.org/abs/1711.05101>`_ 中提出的。

    更新公式如下：

    .. math::
        \begin{array}{ll} \\
            m = \beta_1 * m + (1 - \beta_1) * g \\
            v = \beta_2 * v + (1 - \beta_2) * g * g \\
            update = \frac{m}{\sqrt{v} + \epsilon} \\
            update =
            \begin{cases}
                update + weight\_decay * w
                    & \text{ if } weight\_decay > 0 \\
                update
                    & \text{ otherwise }
            \end{cases} \\
            w  = w - lr * update
        \end{array}

    :math:`m` 代表第一个矩向量，:math:`v` 代表第二个矩向量，:math:`g` 代表 `gradient` ，:math:`\beta_1, \beta_2` 代表 `beta1` 和 `beta2` ，:math:`lr` 代表 `learning_rate` ，:math:`w` 代表 `var` ，:math:`decay` 代表 `weight_decay` ， :math:`\epsilon` 代表 `epsilon` 。

    参数：
        - **use_locking** (bool) - 是否对参数更新加锁保护。如果为True，则 `var` 、 `m` 和 `v` 张量的更新将受到锁的保护。如果为False，则结果不可预测。默认值：False。

    输入：
        - **var** (Parameter) - 需要更新的权重。shape为 :math:`(N, *)` 其中 :math:`*` 表示任何数量的附加维度，数据类型可以是float16或float32。
        - **m** (Parameter) - 更新公式中的第一个动量矩阵，它的shape应该和 `var` 一致，数据类型可以是float16或float32。
        - **v** (Parameter) - 更新公式中的第二个动量矩阵，shape和数据类型与 `m` 相同。
        - **lr** (float) - 更新公式中的 :math:`lr` 。其论文建议值为 :math:`10^{-8}` ，数据类型应为float32。
        - **beta1** (float) - 第一个动量矩阵的指数衰减率，数据类型应为float32。论文建议的值是 :math:`0.9` 。
        - **beta2** (float) - 第二个动量矩阵的指数衰减率，数据类型应为float32。论文建议的值是 :math:`0.999` 。
        - **epsilon** (float) - 添加到分母中的值，以提高数值稳定性，数据类型应为float32。
        - **decay** (float) - 权重衰减值，必须是具有float32数据类型的标量张量。默认值：0.0。
        - **gradient** (Tensor) - 梯度，shape和数据类型与 `var` 相同。

    输出：
        3个张量的Tuple，为更新后的参数。

        - **var** (Tensor) - 具有与 `var` 相同的shape和数据类型。
        - **m** (Tensor) - 具有与 `m` 相同的shape和数据类型。
        - **v** (Tensor) - 具有与 `v` 相同的shape和数据类型。

    异常：
        - **TypeError**: -如果 `use_locking` 不是bool类型。
        - **TypeError**: -如果 `lr`, `beta1`, `beta2`, `epsilon` 或者 `decay` 不是float32。
        - **TypeError**: -如果 `var`, `m` 或者 `v` 不是数据类型为float16或者float32的Parameter。
        - **TypeError**: -如果 `gradient` 不是Tensor。
        - **ValueError** - 如果 `eps` 小于等于0。
        - **ValueError** - 如果 `beta1` 、 `beta2` 不在（0.0,1.0）范围内。
        - **ValueError** - 如果 `decay` 小于0。

mindspore.ops.Adam
==================

.. py:class:: mindspore.ops.Adam(use_locking=False, use_nesterov=False)

    通过Adam算法更新梯度。

    Adam算法详情请参考 `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_ 。

    有关更多详细信息，参见 :class:`mindspore.nn.Adam` 。

    更新公式如下：

    .. math::
        \begin{array}{ll} \\
            m = \beta_1 * m + (1 - \beta_1) * g \\
            v = \beta_2 * v + (1 - \beta_2) * g * g \\
            l = \alpha * \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t} \\
            w = w - l * \frac{m}{\sqrt{v} + \epsilon}
        \end{array}

    其中， :math:`m` 表示第一个动量矩阵， :math:`v` 表示第二个动量矩阵， :math:`g` 表示 `gradient`， :math:`l` 表示缩放因子 `lr` ， :math:`\beta_1, \beta_2` 表示 `beta1` 和 `beta2` ， :math:`t` 表示更新步数， :math:`beta_1^t(\beta_1^{t})` 和 :math:`beta_2^t(\beta_2^{t})` 表示 `beta1_power` 和 `beta2_power` ， :math:`\alpha` 表示 `learning_rate` ， :math:`w` 表示 `var` ， :math:`\epsilon` 表示 `epsilon` 。

    参数：
        - **use_locking** (bool) - 是否对参数更新加锁保护。如果为True，则 `w` 、 `m` 和 `v` 的Tensor更新将受到锁的保护。如果为False，则结果不可预测。默认值：False。
        - **use_nesterov** (bool) - 是否使用Nesterov Accelerated Gradient (NAG)算法更新梯度。如果为True，则使用NAG更新梯度。如果为False，则在不使用NAG的情况下更新梯度。默认值：False。

    输入：
        - **var** (Parameter) - 需更新的权重。shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度，其数据类型可以是float16或float32。
        - **m** (Parameter) - 更新公式中的第一个动量矩阵，shape应与 `var` 相同。
        - **v** (Parameter) - 更新公式中的第二个动量矩阵，shape应与 `var` 相同。
        - **beta1_power** (float) - 在更新公式中的 :math:`beta_1^t(\beta_1^{t})` 。
        - **beta2_power** (float) - 在更新公式中的 :math:`beta_2^t(\beta_2^{t})` 。
        - **lr** (float) - 在更新公式中的 :math:`l` 。其论文建议取值为 :math:`10^{-8}`。
        - **beta1** (float) - 第一个动量矩阵的指数衰减率。论文建议取值为 :math:`0.9` 。
        - **beta2** (float) - 第二个动量矩阵的指数衰减率。论文建议取值为 :math:`0.999` 。
        - **epsilon** (float) - 添加到分母中的值，以提高数值稳定性。
        - **gradient** (Tensor) - 表示梯度，shape和数据类型与 `var` 相同。

    输出：
        3个Tensor的tuple，已更新的参数。

        - **var** (Tensor) - shape和数据类型与输入 `var` 相同。
        - **m** (Tensor) - shape和数据类型与输入 `m` 相同。
        - **v** (Tensor) - shape和数据类型与输入 `v` 相同的。

    异常：
        - **TypeError** - `use_locking` 和 `use_nesterov` 都不是bool。
        - **TypeError** - `var` 、 `m` 或 `v` 不是Parameter。
        - **TypeError** - `beta1_power` 、 `beta2_power1` 、 `lr` 、 `beta1` 、 `beta2` 、 `epsilon` 或 `gradient` 不是Tensor。

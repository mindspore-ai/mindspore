mindspore.ops.ApplyAdagradDA
=============================

.. py:class:: mindspore.ops.ApplyAdagradDA(use_locking=False)

    根据Adagrad算法更新 `var` 。

    Adagrad算法在论文 `Adaptive Subgradient Methods for Online Learning and Stochastic Optimization <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_ 中提出。

    .. math::
        \begin{array}{ll} \\
            grad\_accum += grad \\
            grad\_squared\_accum += grad * grad \\
            tmp\_val=
                \begin{cases}
                     sign(grad\_accum) * max\left \{|grad\_accum|-l1*global\_step, 0\right \} & \text{ if } l1>0 \\
                     grad\_accum & \text{ otherwise } \\
                 \end{cases} \\
            x\_value = -1 * lr * tmp\_val \\
            y\_value = l2 * global\_step * lr + \sqrt{grad\_squared\_accum} \\
            var = \frac{ x\_value }{ y\_value }
        \end{array}

    `var` 、 `gradient_accumulator` 、 `gradient_squared_accumulator` 和 `grad` 的输入遵循隐式类型转换规则，使数据类型一致。
    如果它们具有不同的数据类型，则较低精度的数据类型将转换为相对最高精度的数据类型。

    参数：
        - **use_locking** (bool) - 如果为 ``True`` ， `var` 和 `gradient_accumulator` 的更新将受到锁的保护。否则，行为为未定义，很可能出现较少的冲突。默认值为 ``False`` 。

    输入：
        - **var** (Parameter) - 要更新的变量。数据类型必须为float16或float32。shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。
        - **gradient_accumulator** (Parameter) - 要更新累积的梯度，为公式中的 :math:`grad\_accum` 。shape必须与 `var` 相同。
        - **gradient_squared_accumulator** (Parameter) - 要更新的平方累积的梯度， 为公式中的 :math:`grad\_squared\_accum` 。shape必须与 `var` 相同。
        - **grad** (Tensor) - 梯度，为一个Tensor。shape必须与 `var` 相同。
        - **lr** ([Number, Tensor]) - 学习率。必须是Scalar。数据类型为float32或float16。
        - **l1** ([Number, Tensor]) - L1正则化。必须是Scalar。数据类型为float32或float16。
        - **l2** ([Number, Tensor]) - L2正则化。必须是Scalar。数据类型为float32或float16。
        - **global_step** ([Number, Tensor]) - 训练步骤的编号。必须是Scalar。数据类型为int32或int64。

    输出：
        3个Tensor组成的tuple，更新后的参数。

        - **var** (Tensor) - shape和数据类型与 `var` 相同。
        - **gradient_accumulator** (Tensor) - shape和数据类型与 `gradient_accumulator` 相同。
        - **gradient_squared_accumulator** (Tensor) - shape和数据类型与 `gradient_squared_accumulator` 相同。

    异常：
        - **TypeError** - 如果 `var` 、 `gradient_accumulator` 或 `gradient_squared_accumulator` 不是Parameter。
        - **TypeError** - 如果 `grad` 不是 Tensor。
        - **TypeError** - 如果 `lr` 、 `l1` 、 `l2` 或者 `global_step` 既不是数值型也不是Tensor。
        - **TypeError** - 如果 `use_locking` 不是bool。
        - **TypeError** - 如果 `var` 、 `gradient_accumulator` 、 `gradient_squared_accumulator` 、 `grad` 、 `lr` 、 `l1` 或 `l2` 的数据类型既不是float16也不是float32。 
        - **TypeError** - 如果 `gradient_accumulator` 、 `gradient_squared_accumulator` 、 `grad` 与 `var` 的数据类型不相同。
        - **TypeError** - 如果 `global_step` 的数据类型不是int32也不是int64。
        - **ValueError** - 如果 `lr` 、 `l1` 、 `l2` 和 `global_step` 的shape大小不为0。
        - **TypeError** - 如果 `var` 、 `gradient_accumulator` 、 `gradient_squared_accumulator` 和 `grad` 不支持数据类型转换。

mindspore.ops.SparseApplyAdagradV2
==================================

.. py:class:: mindspore.ops.SparseApplyAdagradV2(lr, epsilon, update_slots=True, use_locking=False)

    根据Adagrad算法更新相关参数。

    .. math::
        \begin{array}{ll} \\
            accum += grad * grad \\
            var -= lr * grad * \frac{1}{\sqrt{accum} + \epsilon}
        \end{array}

    :math:`\epsilon` 代表 `epsilon`。

    `var` 、 `accum` 和 `grad` 的输入遵循隐式类型转换规则，使数据类型一致。如果它们具有不同的数据类型，则低精度数据类型将转换为相对最高精度的数据类型。

    参数：
        - **lr** (float) - 学习率。
        - **epsilon** (float) - 添加到分母上的较小值，以确保数值的稳定性。
        - **update_slots** (bool) - 如果为True，则将更新 `accum` 。默认值：True。
        - **use_locking** (bool) - 是否对参数更新加锁保护。默认值：False。

    输入：
        - **var** (Parameter) - 要更新的变量。为任意维度，其数据类型为float16或float32。
        - **accum** (Parameter) - 要更新的累积。shape和数据类型必须与 `var` 相同。
        - **grad** (Tensor) - 梯度，为一个Tensor。shape和数据类型必须与 `var` 相同，且需要满足 :math:`grad.shape[1:] = var.shape[1:] if var.shape > 1`。
        - **indices** (Tensor) - `var` 和 `accum` 第一维度的索引向量，数据类型为int32，且需要保证 :math:`indices.shape[0] = grad.shape[0]`。

    输出：
        2个Tensor组成的tuple，更新后的参数。

        - **var** (Tensor) - shape和数据类型与 `var` 相同。
        - **accum** (Tensor) - shape和数据类型与 `accum` 相同。

    异常：
        - **TypeError** - 如果 `lr` 或者 `epsilon` 不是float类型。
        - **TypeError** - 如果 `update_slots` 或者 `use_locking` 不是布尔值。
        - **TypeError** - 如果 `var` 、 `accum` 、 `lr` 或 `grad` 的数据类型既不是float16也不是float32。
        - **TypeError** - 如果 `indices` 的数据类型不是int32。
        - **RuntimeError** - 如果 `var` 、 `accum` 和 `grad` 不支持数据类型转换。

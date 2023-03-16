mindspore.ops.SparseApplyProximalAdagrad
=========================================

.. py:class:: mindspore.ops.SparseApplyProximalAdagrad(use_locking=False)

    根据Proximal Adagrad算法更新网络参数。与 :class:`mindspore.ops.ApplyProximalAdagrad` 相比，增加了一个索引Tensor。

    .. math::
        \begin{array}{ll} \\
            accum += grad * grad \\
            \text{prox_v} = var - lr * grad * \frac{1}{\sqrt{accum}} \\
            var = \frac{sign(\text{prox_v})}{1 + lr * l2} * \max(\left| \text{prox_v} \right| - lr * l1, 0)
        \end{array}

    `var` 、 `accum` 和 `grad` 的输入遵循隐式类型转换规则，使数据类型一致。
    如果它们具有不同的数据类型，则较低精度的数据类型将转换为相对最高精度的数据类型。

    参数：
        - **use_locking** (bool) - 如果为True，则将保护 `var` 和 `accum` 参数不被更新。默认值：False。

    输入：
        - **var** (Parameter) - 公式中的"var"。数据类型必须为float16或float32。shape为 :math:`(N, *)` ，其中 :math:`*` 表示任何附加维度。
        - **accum** (Parameter) - 公式中的"accum"，与 `var` 的shape和数据类型相同。
        - **lr** (Union[Number, Tensor]) - 学习率，必须为float或为Tensor，其数据类型为float16或float32。必须大于零。
        - **l1** (Union[Number, Tensor]) - l1正则化，必须为float或为Tensor，其数据类型为float16或float32。必须大于等于零。
        - **l2** (Union[Number, Tensor]) - l2正则化，必须为float或为Tensor，其数据类型为float16或float32。必须大于等于零。
        - **grad** (Tensor) - 梯度，数据类型与 `var` 相同。如果 `var` 的shape大于1，那么 :math:`grad.shape[1:] = var.shape[1:]` 。
        - **indices** (Tensor) - `var` 和 `accum` 第一维度中的索引。如果 `indices` 中存在重复项，则无意义。数据类型必须是int32、int64和 :math:`indices.shape[0] = grad.shape[0]` 。

    输出：
        两个Tensor组成的tuple，更新后的参数。

        - **var** (Tensor) - shape和数据类型与输入 `var` 相同。
        - **accum** (Tensor) - shape和数据类型与输入 `accum` 相同。

    异常：
        - **TypeError** - 如果 `use_locking` 不是bool。
        - **TypeError** - 如果 `var` 、 `accum` 、 `lr` 、 `l1` 、 `l2` 或 `grad` 的数据类型既不是float16也不是float32。
        - **TypeError** - 如果 `indices` 的数据类型既不是int32也不是int64。
        - **ValueError** - 如果 `lr` <= 0 或者 `l1` < 0 或者 `l2` < 0。
        - **RuntimeError** - 如果不支持参数的 `var` 、 `accum` 和 `grad` 数据类型转换。

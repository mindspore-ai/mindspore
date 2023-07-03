mindspore.ops.ApplyAdagrad
===========================

.. py:class:: mindspore.ops.ApplyAdagrad(update_slots=True)

    根据Adagrad算法更新相关参数。

    Adagrad算法在论文 `Adaptive Subgradient Methods for Online Learning and Stochastic Optimization <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_ 中提出。针对不同参数样本数不均匀的问题，自适应的为各个参数分配不同的学习率。

    .. math::
        \begin{array}{ll} \\
            accum += grad * grad \\
            var -= lr * grad * \frac{1}{\sqrt{accum}}
        \end{array}

    `var` 、 `accum` 和 `grad` 的输入遵循隐式类型转换规则，使数据类型一致。如果它们具有不同的数据类型，较低精度的数据类型将转换为相对最高精度的数据类型。

    参数：
        - **update_slots** (bool) - 是否更新 `accum` 参数，如果为 ``True`` ， `accum` 将更新。默认值为： ``True`` 。

    输入：
        - **var** (Parameter) - 要更新的权重。数据类型为float32或float16。shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。
        - **accum** (Parameter) - 要更新的累积。shape必须与 `var` 相同。
        - **lr** (Union[Number, Tensor]) - 学习率，必须是Scalar。数据类型为float32或float16。
        - **grad** (Tensor) - 梯度，为一个Tensor。shape必须与 `var` 相同。

    输出：
        2个Tensor组成的tuple，更新后的数据。

        - **var** (Tensor) - shape和数据类型与 `var` 相同。
        - **accum** (Tensor) - shape和数据类型与 `accum` 相同。

    异常：
        - **TypeError** - 如果 `var` 、 `accum` 、 `lr` 或 `grad` 的数据类型既不是float16也不是float32。
        - **TypeError** - 如果 `lr` 既不是数值型也不是Tensor。
        - **TypeError** - 如果 `var` 、 `accum` 和 `grad` 不支持数据类型转换。

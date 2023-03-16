mindspore.ops.ApplyProximalAdagrad
==================================

.. py:class:: mindspore.ops.ApplyProximalAdagrad(use_locking=False)

    根据Proximal Adagrad算法更新网络参数。
    请参阅论文 `Efficient Learning using Forward-Backward Splitting <http://papers.nips.cc//paper/3793-efficient-learning-using-forward-backward-splitting.pdf>`_ 。

    .. math::
        \begin{array}{ll} \\
            accum += grad * grad \\
            \text{prox_v} = var - lr * grad * \frac{1}{\sqrt{accum}} \\
            var = \frac{sign(\text{prox_v})}{1 + lr * l2} * \max(\left| \text{prox_v} \right| - lr * l1, 0)
        \end{array}

    输入 `var` 、 `accum` 和 `grad` 之间必须遵守隐式类型转换规则以保证数据类型的统一。如果数据类型不同，低精度的数据类型将被自动转换到高精度的数据类型。

    参数：
        - **use_locking** (bool) - 是否对参数更新加锁保护。默认值：False。

    输入：
        - **var** (Parameter) - 公式中的"var"。数据类型需为float16或float32。shape为 :math:`(N, *)` ，其中 :math:`*` 表示任何数量的附加维度。
        - **accum** (Parameter) - 公式中的"accum"。与 `var` 的shape和数据类型相同。
        - **lr** (Union[Number, Tensor]) - 学习率，必须为标量，数据类型为float16或float32。
        - **l1** (Union[Number, Tensor]) - l1正则化强度，必须为标量，数据类型为float16或float32。
        - **l2** (Union[Number, Tensor]) - l2正则化强度，必须为标量，数据类型为float16或float32。
        - **grad** (Tensor) - 梯度，与 `var` 的shape与数据类型相同。

    输出：
        包含两个Tensor的Tuple，已被更新的参数。

        - **var** (Tensor) - 与输入 `var` 的shape与数据类型相同。
        - **accum** (Tensor) - 与输入 `accum` 的shape与数据类型相同。

    异常：
        - **TypeError** - `use_blocking` 不是bool类型。
        - **TypeError** - `var` 、 `lr` 、 `l1` 或 `l2` 的数据类型不是float16或float32。
        - **TypeError** - `lr` 、 `l1` 或 `l2` 的数据类型不是Number或Tensor。
        - **TypeError** - `grad` 不是Tensor。
        - **RuntimeError** - `var` 、 `accum` 和 `grad` 网络参数的数据类型转换错误。

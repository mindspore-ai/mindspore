mindspore.ops.ApplyProximalGradientDescent
==========================================

.. py:class:: mindspore.ops.ApplyProximalGradientDescent

    根据FOBOS(Forward Backward Splitting)算法更新网络参数。
    请参阅论文 `Efficient Learning using Forward-Backward Splitting
    <http://papers.nips.cc//paper/3793-efficient-learning-using-forward-backward-splitting.pdf>`_。

    .. math::
        \begin{array}{ll} \\
            \text{prox_v} = var - \alpha * \delta \\
            var = \frac{sign(\text{prox_v})}{1 + \alpha * l2} * \max(\left| \text{prox_v} \right| - \alpha * l1, 0)
        \end{array}

    其中 :math:`\alpha` 为 `alpha` ，:math:`\delta` 为 `delta` 。

    输入 `var` 和 `delta` 之间必须遵守隐式类型转换规则以保证数据类型的统一。如果数据类型不同，低精度的数据类型将被自动转换到高精度的数据类型。

    输入：
        - **var** (Parameter) - Tensor，公式中的"var"。数据类型为float16或float32。shape为 :math:`(N, *)` ，其中 :math:`*` 表示任何数量的附加维度。
        - **alpha** (Union[Number, Tensor]) - 比例系数，必须为标量。数据类型为float16或float32。
        - **l1** (Union[Number, Tensor]) - l1正则化强度，必须为标量。数据类型为float16或float32。
        - **l2** (Union[Number, Tensor]) - l2正则化强度，必须为标量。数据类型为float16或float32。
        - **delta** (Tensor) - 梯度Tensor。

    输出：
        Tensor，更新后的 `var` 。

    异常：
        - **TypeError** - `var` 、 `alpha` 、 `l1` 或 `l2` 的数据类型非float16或float32。
        - **TypeError** - `alpha` 、 `l1` 或 `l2` 不是Number或Tensor。
        - **TypeError** - `delta` 不是Tensor。
        - **RuntimeError** - `var` 和 `delta` 之间的数值转换不被支持。

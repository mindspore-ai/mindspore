mindspore.ops.SoftmaxCrossEntropyWithLogits
============================================

.. py:class:: mindspore.ops.SoftmaxCrossEntropyWithLogits

    使用one-hot编码获取预测值和真实之间的softmax交叉熵。

    SoftmaxCrossEntropyWithLogits算法的更新公式如下：

    .. math::
        \begin{array}{ll} \\
            p_{ij} = softmax(X_{ij}) = \frac{\exp(x_i)}{\sum_{j = 0}^{N-1}\exp(x_j)} \\
            loss_{ij} = -\sum_j{Y_{ij} * ln(p_{ij})}
        \end{array}

    其中 :math:`X` 代表 `logits` 。 :math:`Y` 代表 `label` 。 :math:`loss` 代表 `output` 。

    输入：
        - **logits** (Tensor) - 输入预测值，其shape为 :math:`(N, C)` ，数据类型为float16或float32。
        - **labels** (Tensor) - 输入真实值，其shape为 :math:`(N, C)` ，数据类型与 `logits` 的相同。

    输出：
        两个Tensor(loss, dlogits)组成的tuple， `loss` 的shape为 :math:`(N,)` ， `dlogits` 的shape与 `logits` 的相同。

    异常：
        - **TypeError** - `logits` 或  `labels` 的数据类型既不是float16也不是float32。
        - **TypeError** - `logits` 或  `labels` 不是Tensor。
        - **ValueError** - `logits` 的shape与 `labels` 的不同。

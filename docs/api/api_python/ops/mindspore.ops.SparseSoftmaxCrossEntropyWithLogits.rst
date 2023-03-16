mindspore.ops.SparseSoftmaxCrossEntropyWithLogits
==================================================

.. py:class:: mindspore.ops.SparseSoftmaxCrossEntropyWithLogits(is_grad=False)

    计算预测值和标签之间的稀疏softmax交叉熵。

    将预测值设置为 `X` ，输入标签设置为 `Y` ，输出设置为 `loss` 。然后，

    .. math::
        \begin{array}{ll} \\
            p_{ij} = softmax(X_{ij}) = \frac{\exp(x_i)}{\sum_{j = 0}^{N-1}\exp(x_j)} \\
            loss_{ij} = \begin{cases} -ln(p_{ij}), &j = y_i \cr 0, & j \neq y_i \end{cases} \\
            loss = \sum_{ij} loss_{ij}
        \end{array}

    参数：
        - **is_grad** (bool) - 如果为True，则返回计算的梯度。默认值：False。

    输入：
        - **logits** (Tensor) - 输入的预测值，其shape为 :math:`(N, C)` 。数据类型必须为float16或float32。
        - **labels** (Tensor) - 输入的标签，其shape为 :math:`(N)` 。数据类型必须为int32或int64。

    输出：
        Tensor，如果 `is_grad` 为False，则输出Tensor是损失值，是一个Tensor；如果 `is_grad` 为True，则输出记录的是输入的梯度，其shape与 `logits` 相同。

    异常：
        - **TypeError** - 如果 `is_grad` 不是bool。
        - **TypeError** - 如果 `logits` 的数据类型既不是float16也不是float32。
        - **TypeError** - 如果 `labels` 的数据类型既不是int32也不是int64。
        - **ValueError** - 如果 :math:`logits.shape[0] != labels.shape[0]` 。

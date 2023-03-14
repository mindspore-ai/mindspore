mindspore.nn.BCEWithLogitsLoss
===============================

.. py:class:: mindspore.nn.BCEWithLogitsLoss(reduction='mean', weight=None, pos_weight=None)

    输入经过sigmoid激活函数后作为预测值，`BCEWithLogitsLoss` 计算预测值和目标值之间的二值交叉熵损失。

    将输入 `logits` 设置为 :math:`X`，输入 `labels` 为 :math:`Y`，输出为 :math:`L`。则公式如下：

    .. math::
        p_{ij} = sigmoid(X_{ij}) = \frac{1}{1 + e^{-X_{ij}}}

    .. math::
        L_{ij} = -[Y_{ij} \cdot log(p_{ij}) + (1 - Y_{ij}) \cdot log(1 - p_{ij})]

    然后，

    .. math::
        \ell(x, y) = \begin{cases}
        L, & \text{if reduction} = \text{'none';}\\
        \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
        \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    参数：
        - **reduction** (str) - 指定输出结果的计算方式。可选值有：'mean' ， 'sum' ，和 'none' 。如果为 'none' ，则不执行reduction。默认值：'mean' 。
        - **weight** (Tensor, 可选) - 指定每个批次二值交叉熵的权重。如果不是None，将进行广播，其shape与 `logits` 的shape保持一致，数据类型为float16或float32。默认值：None。
        - **pos_weight** (Tensor, 可选) - 指定正样本的权重。是一个长度等于分类数的向量。如果不是None，将进行广播，其shape与 `logits` 的shape保持一致，数据类型必须为float16或float32。默认值：None。

    输入：
        - **logits** (Tensor) - 输入预测值Tensor，shape :math:`(N, *)` ，其中 :math:`*` 代表任意数量的附加维度。数据类型必须为float16或float32。
        - **labels** (Tensor) - 输入目标值Tensor，shape :math:`(N, *)` ，其中 :math:`*` 代表任意数量的附加维度。与 `logits` 的shape和数据类型相同。

    输出：
        Tensor或Scalar，如果 `reduction` 为 'none'，其shape需和 `logits` 相同。否则，将返回Scalar。

    异常：
        - **TypeError** - `logits` 或 `labels` 的不为Tensor。
        - **TypeError** - `logits` 或 `labels` 的数据类型既不是float16也不是float32。
        - **TypeError** - `weight` 或 `pos_weight` 是 Parameter。
        - **TypeError** - `weight` 或 `pos_weight` 的数据类型既不是float16也不是float32。
        - **TypeError** - `reduction` 的数据类型不是string。
        - **ValueError** - `weight` 或 `pos_weight` 不能广播到shape为 `logits` 的Tensor。
        - **ValueError** - `reduction` 不为 'none'、'mean' 或 'sum'。



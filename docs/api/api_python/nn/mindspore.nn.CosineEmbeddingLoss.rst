mindspore.nn.CosineEmbeddingLoss
=================================

.. py:class:: mindspore.nn.CosineEmbeddingLoss(margin=0.0, reduction='mean')

    余弦相似度损失函数，用于测量两个Tensor之间的相似性。

    给定两个Tensor :math:`x1` 和 :math:`x2` ，以及一个Tensor标签 :math:`y` ，值为1或-1，公式如下：

    .. math::
        loss(x_1, x_2, y) = \begin{cases}
        1-cos(x_1, x_2), & \text{if } y = 1\\
        max(0, cos(x_1, x_2)-margin), & \text{if } y = -1\\
        \end{cases}

    参数：
        - **margin** (float) - 指定运算的调节因子，取值范围[-1.0, 1.0]。默认值：0.0。
        - **reduction** (str) - 指定输出结果的计算方式。可选值为"none"、"mean"或"sum"，分别表示不指定计算方式、使用均值计算和使用求和计算。默认值："mean"。

    输入：
        - **logits_x1** (Tensor) - 输入Tensor，shape :math:`(N, *)` ，其中 :math:`*` 代表任意数量的附加维度。
        - **logits_x2** (Tensor) - 输入Tensor，shape :math:`(N, *)` 。shape和数据类型与 `logits_x1` 相同。
        - **labels** (Tensor) - 输入值为1或-1。假设 `logits_x1` 的shape是 :math:`(x_1, x_2, x_3, ..., x_R)` ，那么 `labels` 的shape必须是 :math:`(x_1, x_3, x_4, ..., x_R)` 。

    输出：
        Tensor或Scalar，如果 `reduction` 为"none"，其shape与 `labels` 相同。否则，将返回为Scalar。

    异常：
        - **TypeError** - `margin` 不是float。
        - **ValueError** - `reduction` 不为"none"、"mean"或"sum"。
        - **ValueError** - `margin` 的值不在范围[-1.0, 1.0]内。



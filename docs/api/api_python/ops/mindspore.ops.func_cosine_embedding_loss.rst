mindspore.ops.cosine_embedding_loss
===================================

.. py:function:: mindspore.ops.cosine_embedding_loss(input1, input2, target, margin=0.0, reduction="mean")

    余弦相似度损失函数，用于测量两个Tensor之间的相似性。

    给定两个Tensor :math:`input1` 和 :math:`input2` ，以及一个Tensor标签 :math:`target` ，值为1或-1，公式如下：

    .. math::
        loss(input1, input2, target) = \begin{cases}
        1-cos(input1, input2), & \text{if } target = 1\\
        max(0, cos(input1, input2)-margin), & \text{if } target = -1\\
        \end{cases}

    参数：
        - **input1** (Tensor) - 输入Tensor，shape :math:`(N, *)` ，其中 :math:`*` 代表任意数量的附加维度。
        - **input2** (Tensor) - 输入Tensor，shape :math:`(N, *)` 。shape和数据类型与 `input1` 相同。
        - **target** (Tensor) - 输入值为1或-1。假设 `input1` 的shape是 :math:`(x_1, x_2, x_3, ..., x_R)` ，那么 `target` 的shape必须是 :math:`(x_1, x_3, x_4, ..., x_R)` 。
        - **margin** (float，可选) - 指定运算的调节因子，取值范围[-1.0, 1.0]。默认值： ``0.0`` 。
        - **reduction** (str，可选) - 指定应用于输出结果的规约计算方式，可选 ``'none'`` 、 ``'mean'`` 、 ``'sum'`` ，默认值： ``'mean'`` 。

          - ``"none"``：不应用规约方法。
          - ``"mean"``：计算输出元素的平均值。
          - ``"sum"``：计算输出元素的总和。

    返回：
        Tensor或Scalar，如果 `reduction` 为"none"，其shape与 `target` 相同。否则，将返回为Scalar。

    异常：
        - **TypeError** - `margin` 不是float。
        - **ValueError** - `reduction` 不为"none"、"mean"或"sum"。
        - **ValueError** - `margin` 的值不在范围[-1.0, 1.0]内。



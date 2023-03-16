mindspore.nn.MarginRankingLoss
===============================

.. py:class:: mindspore.nn.MarginRankingLoss(margin=0.0, reduction='mean')

    排序损失函数，用于创建一个衡量给定损失的标准。
    
    给定两个Tensor :math:`input1` 和 :math:`input2` ，以及一个Tensor标签 :math:`target` ，值为1或-1，公式如下：
    
    .. math::
        \text{loss}(input1, input2, target) = \max(0, -target * (input1 - input2) + \text{margin})

    参数：
        - **margin** (float) - 指定运算的调节因子。默认值：0.0。
        - **reduction** (str) - 指定输出结果的计算方式。可选值为"none"、"mean"或"sum"，分别表示不指定计算方式、使用均值计算和使用求和计算。默认值："mean"。

    输入：
        - **input1** (Tensor) - 输入Tensor，shape :math:`(N, *)` ，其中 :math:`*` 代表任意数量的附加维度。
        - **input2** (Tensor) - 输入Tensor，shape :math:`(N, *)` 。shape和数据类型与 `input1` 相同。
        - **target** (Tensor) - 输入值为1或-1。假设 `input1` 的shape是 :math:`(x_1, x_2, x_3, ..., x_R)` ，那么 `target` 的shape必须是 :math:`(x_1, x_2, x_3, ..., x_R)` 。

    输出：
        Tensor或Scalar，如果 `reduction` 为"none"，其shape与 `labels` 相同。否则，将返回为Scalar。

    异常：
        - **TypeError** - `margin` 不是float。
        - **TypeError** - `input1` ，`input2` 和 `target` 不是Tensor。
        - **TypeError** - `input1` 和 `input2` 类型不一致。
        - **TypeError** - `input1` 和 `target` 类型不一致。
        - **ValueError** - `input1` 和 `input2` shape不一致。
        - **ValueError** - `input1` 和 `target` shape不一致。
        - **ValueError** - `reduction` 不为"none"，"mean"或"sum"。

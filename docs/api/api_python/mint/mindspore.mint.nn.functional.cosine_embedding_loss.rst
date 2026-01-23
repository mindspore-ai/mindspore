mindspore.mint.nn.functional.cosine_embedding_loss
===================================================

.. py:function:: mindspore.mint.nn.functional.cosine_embedding_loss(input1, input2, target, margin=0.0, reduction="mean")

    余弦相似度损失函数，用于测量两个Tensor之间的相似性。

    给定两个Tensor :math:`x1` 和 :math:`x2` ，以及一个Tensor标签 :math:`y` （正样本的值为1，负样本的值为-1），公式如下：

    .. math::
        loss(x_1, x_2, y) = \begin{cases}
        1-cos(x_1, x_2), & \text{if } y = 1\\
        \max(0, cos(x_1, x_2)-margin), & \text{if } y = -1\\
        \end{cases}


    参数：
        - **input1** (Tensor) - 输入Tensor，shape为 :math:`(N, D)` 或 :math:`(D)` ，其中 :math:`N` 代表批量大小，:math:`D` 代表嵌入维度。
        - **input2** (Tensor) - 输入Tensor，shape为 :math:`(N, D)` 或 :math:`(D)` 。数据类型与 `input1` 相同，shape需与 `input1` 一致或满足广播规则。
        - **target** (Tensor) - 标签Tensor，输入值为1或-1。shape为 :math:`(N)` 或 :math:`()` 。
        - **margin** (float，可选) - 指定负样本运算中的调节因子，取值范围[-1.0, 1.0]，范围外的取值不会报错，但无实际意义。默认值： ``0.0`` 。
        - **reduction** (str，可选) - 指定应用于输出结果的归约计算方式，可选 ``"none"`` 、 ``"mean"`` 、 ``"sum"`` ，默认值： ``"mean"`` 。

          - ``"none"`` ：不应用归约方法。
          - ``"mean"`` ：计算输出元素的平均值。
          - ``"sum"`` ：计算输出元素的总和。

    返回：
        Tensor或Scalar。如果 `reduction` 为 ``"none"`` ，返回一个shape与 `target` 相同的Tensor；否则，将返回一个Scalar。

    异常：
        - **ValueError** - `reduction` 不为 ``"none"`` 、 ``"mean"`` 或 ``"sum"`` 。
        - **ValueError** - `input1` 和 `input2` 的形状不匹配。
        - **ValueError** - `target` 的形状和 `input1` 及 `input2` 的形状不匹配。

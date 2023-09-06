mindspore.nn.HingeEmbeddingLoss
===============================

.. py:class:: mindspore.nn.HingeEmbeddingLoss(margin=1.0, reduction="mean")

    Hinge Embedding 损失函数。根据输入 `logits` 和 `labels` （只包含1或-1）计算Hinge Embedding损失值。通常被用来衡量两个输入之间的相似度。

    mini-batch中的第 :math:`n` 个样例的损失函数为：

    .. math::
        l_n = \begin{cases}
        x_n, & \text{if}\; y_n = 1,\\
        \max \{0, \Delta - x_n\}, & \text{if}\; y_n = -1,
        \end{cases}

    总损失值为：

    .. math::
        \ell(x, y) = \begin{cases}
        \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
        \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    其中 :math:`L = \{l_1,\dots,l_N\}^\top`。

    参数：
        - **margin** (float, int) - Hinge Embedding Loss公式定义的阈值 :math:`margin`。公式中表示为 :math:`\Delta`。默认值： ``1.0``。
        - **reduction** (str，可选) - 指定应用于输出结果的规约计算方式，可选 ``'none'`` 、 ``'mean'`` 、 ``'sum'`` ，默认值： ``'mean'`` 。

          - ``"none"``：不应用规约方法。
          - ``"mean"``：计算输出元素的平均值。
          - ``"sum"``：计算输出元素的总和。

    输入：
        - **logits** (Tensor) - 预测值，公式中表示为 :math:`x`，shape为 :math:`(*)`。 :math:`*` 代表着任意数量的维度。
        - **labels** (Tensor) - 标签值，公式中表示为 :math:`y`，和 `logits` 具有相同shape，包含1或-1。

    返回：
        Tensor或Tensor scalar，根据 :math:`reduction` 计算的loss。

    异常：
        - **TypeError** - `logits` 不是Tensor。
        - **TypeError** - `labels` 不是Tensor。
        - **TypeError** - `margin` 不是float或int。
        - **ValueError** - `labels` 和 `logits` shape不一致且不能广播。
        - **ValueError** - `reduction` 不是 ``"none"`` 、 ``"mean"`` 或者 ``"sum"`` 。

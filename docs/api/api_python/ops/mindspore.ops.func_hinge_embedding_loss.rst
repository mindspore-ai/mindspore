mindspore.ops.hinge_embedding_loss
===================================

.. py:function:: mindspore.ops.hinge_embedding_loss(inputs, targets, margin=1.0, reduction="mean")

    Hinge Embedding 损失函数，衡量输入 `inputs` 和标签 `targets` （包含1或-1）之间的损失值。

    mini-batch中的第n个样例的损失函数为：

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
        - **inputs** (Tensor) - 预测值，公式中表示为 :math:`x`，shape为 :math:`(*)`。`*` 代表着任意数量的维度。
        - **targets** (Tensor) - 标签值，公式中表示为 :math:`y`，和 `logits` 具有相同shape，包含1或-1。
        - **margin** (float, int) - Hinge Embedding Loss公式定义的阈值 :math:`margin`。公式中表示为 :math:`\Delta`。默认值：1.0。
        - **reduction** (str) - 指定应用于输出结果的计算方式，'none'、'mean'、'sum'，默认值：'mean'。

    返回：
        Tensor或Tensor scalar，根据 :math:`reduction` 计算的loss。

    异常：
        - **TypeError** - `inputs` 不是数据类型为float的Tensor。
        - **TypeError** - `targets` 不是数据类型为float的Tensor。
        - **TypeError** - `margin` 不是float或者int。
        - **ValueError** - `inputs` 和 `targets` shape不一致。
        - **ValueError** - `reduction` 不是"none"、"mean"或者"sum"。

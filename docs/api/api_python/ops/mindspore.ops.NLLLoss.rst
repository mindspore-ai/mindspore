mindspore.ops.NLLLoss
======================

.. py:class:: mindspore.ops.NLLLoss(reduction="mean")

    获取预测值和目标值之间的负对数似然损失。

    reduction为'none'时，负对数似然损失如下：

    .. math::
        \ell(x, t)=L=\left\{l_{1}, \ldots, l_{N}\right\}^{\top},
        \quad l_{n}=-w_{t_{n}} x_{n, t_{n}},
        \quad w_{c}=\text { weight }[c] \cdot 1

    其中， :math:`x` 表示预测值， :math:`t` 表示目标值， :math:`w` 表示权重，N表示batch size， :math:`c` 限定范围为[0, C-1]，表示类索引，其中 :math:`C` 表示类的数量。

    reduction不为'none'（默认为'mean'），则

    .. math::
        \ell(x, t)=\left\{\begin{array}{ll}
        \sum_{n=1}^{N} \frac{1}{\sum_{n=1}^{N} w_{t n}} l_{n}, & \text { if reduction }=\text { 'mean'; } \\
        \sum_{n=1}^{N} l_{n}, & \text { if reduction }=\text { 'sum' }
        \end{array}\right.

    参数：
        - **reduction** (str) - 指定应用于输出结果的计算方式，比如'none'、'mean'，'sum'，默认值："mean"。

    输入：
        - **logits** (Tensor) - 输入预测值，shape为 :math:`(N, C)` 。数据类型仅支持float32或float16。
        - **labels** (Tensor) - 输入目标值，shape为 :math:`(N,)` ，取值范围为 :math:`[0, C-1]` 。数据类型仅支持int32或int64。
        - **weight** (Tensor) - 指定各类别的权重，shape为 :math:`(C,)` ，数据类型仅支持float32或float16。

    输出：
        由 `loss` 和 `total_weight` 组成的2个Tensor的tuple。

        - **loss** (Tensor) - 当 `reduction` 为'none'且 `logits` 为二维Tensor时， `loss` 的shape为 :math:`(N,)` 。否则， `loss` 为scalar。数据类型与 `logits` 相同。
        - **total_weight** (Tensor) - `total_weight` 是scalar，数据类型与 `weight` 相同。

    异常：
        - **TypeError** - `logits` 或 `weight` 的数据类型既不是float16也不是float32。
        - **TypeError** - `labels` 的数据类型既不是int32也不是int64。
        - **ValueError** - `logits` 不是二维Tensor， `labels` 和 `weight` 不是一维Tensor。 `logits` 的第一个维度不等于 `labels` ， `logits` 的第二个维度不等于 `weight` 。
        - **ValueError** - `labels` 的取值超出 :math:`[0, C-1]` ，其中 :math:`C` 表示类的数量。